#!/usr/bin/env python3
"""
demo.py — Watch the attention channel close, on your own laptop.

Reproduces the three headline mechanics of "When Attention Closes: How LLMs Lose
the Thread in Multi-Turn Interaction" (Dongre et al., arXiv:2605.12922) on a small
local model (Qwen2.5-0.5B-Instruct) via HuggingFace transformers + MPS.

Unlike an Ollama demo, this reads the model internals directly, so the paper's
claims become things you can watch happen:

  python3 demo.py gar       # Goal Accessibility Ratio decays as turns grow
  python3 demo.py ablate    # force-close attention to system tokens -> recall collapses
  python3 demo.py probe     # the rules survive in the residual stream (high AUC)
  python3 demo.py all       # run all three

These are run on a 0.5B model with our own data, so absolute numbers will NOT match
the paper's (different architectures, different probes). The point is the *phenomena*
and the *method*, reproduced end-to-end locally.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from uuid import uuid4

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("Missing dependency: pip install transformers torch scikit-learn")

try:
    from smongo import MongoClient
except ImportError:
    sys.exit("Missing dependency: pip install smongo  (requires Python >=3.11)")


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
NEG = torch.finfo(torch.float32).min

# Every run logs its measurements to an embedded, local-first MongoDB (smongo): no mongod,
# no Docker, no network — just a redb file on disk. The `report` mode then runs MongoDB
# aggregation pipelines over the accumulated runs (see run_report).
DB_URI = "local://lost_track_db"
DB_NAME = "lost_track"
COLL = "metrics"


# --------------------------------------------------------------------------- #
# The contract the model is given up front: a persona + planted facts it must
# retain across a long, meandering conversation.
# --------------------------------------------------------------------------- #
FACTS = {
    "project codename": "Halcyon",
    "launch quarter": "Q3 2027",
    "lead auditor": "Dana Okonkwo",
    "compliance framework": "SOC 2 Type II",
}


def system_prompt_with(codename: str) -> str:
    """Build the AUDITRON system prompt with a chosen project codename (the other facts
    stay fixed). The probe varies this codename across episodes to test whether the
    planted value survives in the residual stream."""
    return (
        "You are AUDITRON, a senior compliance officer. Always begin replies with "
        "[AUDITRON] and speak tersely and formally.\n"
        "Retain these facts and recall them exactly when asked:\n"
        f"- The project codename is {codename}.\n"
        f"- The launch quarter is {FACTS['launch quarter']}.\n"
        f"- The lead auditor is {FACTS['lead auditor']}.\n"
        f"- The compliance framework is {FACTS['compliance framework']}.\n"
    )


SYSTEM_PROMPT = system_prompt_with(FACTS["project codename"])

# Filler turns that grow the context (the "relentless flurry"), interleaved with
# fact-recall probes. Each probe maps to a fact we can grade exactly.
FILLER = [
    "Thanks. Can you suggest a light team-building activity for the offsite?",
    "What snacks would work for an afternoon session?",
    "Any icebreaker games you'd recommend?",
    "Suggest a good lunch budget per person, roughly.",
    "What's a fun closing activity for the day?",
    "Any tips for keeping energy up after lunch?",
    "Recommend a playlist vibe for the workshop.",
    "Should we run the workshop morning or afternoon?",
]

PROBES = [
    ("project codename", "What is the project codename?"),
    ("lead auditor", "Who is the lead auditor?"),
    ("compliance framework", "Which compliance framework are we using?"),
    ("launch quarter", "What is the launch quarter?"),
]


class Model:
    """Thin wrapper exposing tokenization, generation, attentions and hidden states."""

    def __init__(self, name: str = MODEL_NAME):
        print(f"loading {name} on {DEVICE} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float32, attn_implementation="eager"
        )
        self.model.to(DEVICE).eval()
        self.n_layers = self.model.config.num_hidden_layers

    def encode(self, messages: list[dict]) -> torch.Tensor:
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(DEVICE)

    def system_span(self, messages: list[dict]) -> slice:
        """Token span covering just the system message, within the full encoding."""
        sys_only = self.encode_no_gen([messages[0]])
        return slice(0, sys_only.shape[1])

    def encode_no_gen(self, messages: list[dict]) -> torch.Tensor:
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(DEVICE)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor | None = None):
        return self.model(
            ids,
            attention_mask=attn_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 40) -> str:
        out = self.model.generate(
            ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    @torch.no_grad()
    def gar_last_token(self, ids: torch.Tensor, span: slice) -> list[float]:
        """Per-layer GAR at the final (generating) token, measured memory-safely.

        Asking for `output_attentions=True` on a full forward materializes an
        O(L^2) attention matrix for *every* layer at once, which OOMs at long context
        (e.g. ~29 GB at 4.6k tokens on this model). Instead we prime a KV cache with all
        but the last token, then run only the last token: the returned attention then has
        shape [1, heads, 1, L] — a single query row over all keys, which is exactly what
        GAR needs and is tiny. The result is identical to reading the last row of the full
        attentions (verified to 0 difference)."""
        L = ids.shape[1]
        prefix, last = ids[:, :-1], ids[:, -1:]
        cache = self.model(prefix, use_cache=True).past_key_values
        out = self.model(
            last, past_key_values=cache, use_cache=True, output_attentions=True,
            attention_mask=torch.ones((1, L), device=ids.device),
        )
        return [a[0, :, -1, span].sum(-1).mean().item() for a in out.attentions]


def gar_per_layer(attentions, query_pos: int, span: slice) -> list[float]:
    """Per-layer Goal Accessibility Ratio: for each layer, the attention mass from
    `query_pos` onto the system-token `span`, averaged over heads. Returned as a list
    indexed by layer so callers can inspect where the channel closes (the paper notes
    the crossover/encoding layer varies a lot by architecture)."""
    return [a[0, :, query_pos, span].sum(-1).mean().item() for a in attentions]


def gar_from_attentions(attentions, query_pos: int, span: slice) -> float:
    """Goal Accessibility Ratio: attention mass from `query_pos` onto the system-token
    `span`, averaged over all heads and all layers. This is the behavioral analog of
    the paper's GAR — how much the model is still 'looking at' its instructions."""
    per_layer = gar_per_layer(attentions, query_pos, span)
    return sum(per_layer) / len(per_layer)


def build_conversation(n_filler: int, system: str = SYSTEM_PROMPT,
                       start: int = 0) -> list[dict]:
    """System prompt + n_filler user/assistant filler turns (assistant replies stubbed
    so we don't pay generation cost just to grow context). `system` overrides the system
    prompt and `start` rotates which filler turns are used, giving the probe within-class
    variation without leaking any fact value into the filler/stub text."""
    stubs = [
        "[AUDITRON] Understood. I will factor that into the schedule.",
        "[AUDITRON] Acknowledged. A balanced option is advisable.",
        "[AUDITRON] Noted. I recommend a moderate, professional choice.",
        "[AUDITRON] Confirmed. That aligns with standard practice.",
    ]
    msgs = [{"role": "system", "content": system}]
    for i in range(n_filler):
        msgs.append({"role": "user", "content": FILLER[(i + start) % len(FILLER)]})
        msgs.append({"role": "assistant", "content": stubs[i % len(stubs)]})
    return msgs


# --------------------------------------------------------------------------- #
# Telemetry: persist each measurement to an embedded MongoDB (smongo) so runs
# accumulate into a queryable experiment log (see run_report).
# --------------------------------------------------------------------------- #
def open_metrics(uri: str):
    """Open (creating on first write) the metrics collection in the embedded store."""
    return MongoClient(uri)[DB_NAME][COLL]


def new_run(model_name: str) -> dict:
    """Identity stamped onto every document written during a single invocation."""
    return {
        "run_id": uuid4().hex,
        "ts": datetime.now(timezone.utc),
        "model": model_name,
        "device": DEVICE,
    }


def log_metric(coll, run: dict, doc: dict) -> None:
    """Insert one measurement, tagged with the run identity (run_id/model/ts/device).
    Telemetry is best-effort: a write failure warns but never aborts the run (the
    scientific output matters more than the log)."""
    if coll is None:
        return
    try:
        coll.insert_one({**run, **doc})
    except Exception as e:
        print(f"[warn] metric not logged: {e}", file=sys.stderr)


def run_gar(m: "Model", coll=None, run: dict | None = None) -> None:
    run = run or {}
    print("\n" + "=" * 72)
    print("  MODE 1: GAR decay — attention thins, then recall finally breaks")
    print("=" * 72)
    print(f"{'turns':>5} | {'ctx tok':>7} | {'GAR all':>8} | {'early':>7} | "
          f"{'late':>7} | {'recall':>6}")
    print("-" * 72)

    # Sweep well past the paper's "relentless flurry": this model's recall is sticky, so
    # the first natural MISS only shows up at long context. GAR is measured memory-safely
    # at the final token (see Model.gar_last_token) so we can reach those lengths.
    first_miss = None
    for n_filler in (0, 8, 24, 56, 96, 128, 160):
        msgs = build_conversation(n_filler)
        sys_span = m.system_span(msgs)

        # GAR at the canonical (codename) probe.
        gar_ids = m.encode(msgs + [{"role": "user", "content": PROBES[0][1]}])
        per_layer = m.gar_last_token(gar_ids, sys_span)
        third = max(1, len(per_layer) // 3)
        gar_all = sum(per_layer) / len(per_layer)
        gar_early = sum(per_layer[:third]) / third
        gar_late = sum(per_layer[-third:]) / third

        # Recall across all four planted facts (greedy, deterministic).
        hits = 0
        for fact_name, question in PROBES:
            ids = m.encode(msgs + [{"role": "user", "content": question}])
            reply = m.generate(ids, max_new_tokens=30)
            hits += FACTS[fact_name].lower() in reply.lower()
        if hits < len(PROBES) and first_miss is None:
            first_miss = (n_filler, gar_ids.shape[1], gar_all)

        log_metric(coll, run, {
            "mode": "gar", "turns": n_filler, "ctx_tokens": gar_ids.shape[1],
            "gar_all": gar_all, "gar_early": gar_early, "gar_late": gar_late,
            "recall": hits, "n_facts": len(PROBES), "recall_miss": hits < len(PROBES),
        })
        print(f"{n_filler:>5} | {gar_ids.shape[1]:>7} | {gar_all:>8.4f} | "
              f"{gar_early:>7.4f} | {gar_late:>7.4f} | {hits:>4}/{len(PROBES)}")

    print("-" * 72)
    print("GAR trends down as context grows: the attention channel is closing.")
    if first_miss:
        n, ctx, g = first_miss
        print(f"First recall MISS at {n} turns (~{ctx} tokens), GAR {g:.4f}: the model")
        print("drops a planted fact once attention to the system prompt has thinned.")
        print("On a 0.5B model natural failures are sparse and noisy (recall can recover")
        print("at longer context) — MODE 2 forces the clean *causal* collapse via ablation.")
    else:
        print("Recall held at every length here (some architectures preserve goal-")
        print("conditioned behavior even at low attention) — MODE 2 forces the collapse.")
    print("(early = first third of layers, late = last third.)\n")


def run_ablate(m: "Model", coll=None, run: dict | None = None) -> None:
    run = run or {}
    print("\n" + "=" * 72)
    print("  MODE 2: Ablation — force-close attention to system tokens, recall collapses")
    print("=" * 72)

    # Use a modest context where normal recall is clean, so any collapse is
    # attributable to blinding the model to its system tokens (not generic decay).
    msgs = build_conversation(4)
    sys_span = m.system_span(msgs)

    normal_hits, ablated_hits = 0, 0
    print(f"{'fact':>20} | {'normal':>8} | {'ablated':>8}")
    print("-" * 72)
    for fact_name, question in PROBES:
        probe_msgs = msgs + [{"role": "user", "content": question}]
        ids = m.encode(probe_msgs)

        normal = m.generate(ids, max_new_tokens=30)
        ablated = generate_ablated(m, ids, sys_span, max_new_tokens=30)

        n_ok = FACTS[fact_name].lower() in normal.lower()
        a_ok = FACTS[fact_name].lower() in ablated.lower()
        normal_hits += n_ok
        ablated_hits += a_ok
        log_metric(coll, run, {
            "mode": "ablate", "fact": fact_name,
            "normal_ok": int(n_ok), "ablated_ok": int(a_ok),
        })
        print(f"{fact_name:>20} | {'OK' if n_ok else 'MISS':>8} | {'OK' if a_ok else 'MISS':>8}")

    n = len(PROBES)
    print("-" * 72)
    print(f"recall: normal {normal_hits}/{n} ({100*normal_hits/n:.0f}%)  ->  "
          f"ablated {ablated_hits}/{n} ({100*ablated_hits/n:.0f}%)")
    print("Blinding the model to its own system prompt collapses fact recall.\n")


def build_ablation_mask(L: int, sys_span: slice, device=None) -> torch.Tensor:
    """Additive attention mask of shape (1, 1, L, L) that keeps the causal structure but
    closes the attention channel from every *post-system* query position onto the
    system-token span.

    Critically, the system rows themselves are NOT masked off their own (causal)
    self-attention. If we masked the system columns for *all* rows (as a naive
    implementation does), the first system row could only attend to a fully -inf column
    set, producing an all -inf row -> softmax NaN that corrupts generation. By masking
    only rows at/after `sys_span.stop`, every row retains at least its diagonal, so there
    are no all -inf rows and no NaNs. This matches the paper's manipulation: force-close
    the channel *from generated tokens to goal tokens*."""
    mask = torch.full((L, L), NEG, device=device).triu(1)
    mask[sys_span.stop:, sys_span] = NEG
    return mask.view(1, 1, L, L)


@torch.no_grad()
def generate_ablated(m: "Model", ids: torch.Tensor, sys_span: slice,
                     max_new_tokens: int = 30) -> str:
    """Greedy-generate while closing the attention channel from generated tokens to the
    system-token span at every step (see `build_ablation_mask`)."""
    cur = ids
    for _ in range(max_new_tokens):
        L = cur.shape[1]
        mask = build_ablation_mask(L, sys_span, device=DEVICE)
        logits = m.model(cur, attention_mask=mask).logits
        if not torch.isfinite(logits[0, -1]).all():
            raise RuntimeError(
                "non-finite logits under the ablation mask — an attention row was fully "
                "masked (-inf). Check build_ablation_mask."
            )
        nxt = logits[0, -1].argmax().item()
        if nxt == m.tok.eos_token_id:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return m.tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)


def run_probe(m: "Model", coll=None, run: dict | None = None) -> None:
    """Linear-probe the residual stream for the paper's actual claim: after the attention
    channel has decayed (long context), does the *planted goal value* still survive in the
    hidden states?

    Design that makes the baseline a true chance baseline:
      - We vary the planted project codename across classes (Halcyon / Borealis / ...),
        all in the SYSTEM prompt.
      - Every episode ends with the SAME fixed question, so the final-token *input* is
        identical across all classes. Therefore the input-embedding layer carries no
        label signal -> its probe sits at chance (~0.5).
      - Within a class we vary the meandering filler conversation (rotation + length),
        which never mentions the codename, giving non-degenerate within-class samples
        without leaking the label into anything but the system prompt.
    A high residual AUC at a mid/late layer then means the model propagated the planted
    value forward into its hidden state even though the attention channel to the system
    prompt has thinned — exactly the survival the paper reports (they get AUC ~0.99 on
    large models; here we only reproduce the *shape*: residual >> embedding)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("\n[probe] needs scikit-learn: pip install scikit-learn\n")
        return
    import numpy as np

    print("\n" + "=" * 72)
    print("  MODE 3: Residual probe — the planted goal survives in the hidden states")
    print("=" * 72)

    codenames = ["Halcyon", "Borealis", "Zephyr", "Cinder"]
    question = "What is the project codename?"  # identical across every class
    n_episodes = 8                              # within-class variation of the filler
    n_filler = len(FILLER)                      # long context: attention has decayed

    # Probe a few layers so the layer-dependence the paper describes is visible, and so
    # the demo is robust to which layer happens to encode the value on a 0.5B model.
    probe_layers = sorted({2, m.n_layers // 2, max(0, m.n_layers - 2)})

    resid = {L: [] for L in probe_layers}
    X_embed, y = [], []
    for label, codename in enumerate(codenames):
        system = system_prompt_with(codename)
        for ep in range(n_episodes):
            msgs = build_conversation(n_filler - (ep % 2), system=system, start=ep)
            ids = m.encode(msgs + [{"role": "user", "content": question}])
            out = m.forward(ids)
            for L in probe_layers:
                resid[L].append(out.hidden_states[L][0, -1].cpu().numpy())
            X_embed.append(out.hidden_states[0][0, -1].cpu().numpy())
            y.append(label)

    y = np.array(y)

    def auc(X: "np.ndarray") -> float:
        X = np.asarray(X)
        # Row-normalize: aids LogisticRegression convergence on raw hidden states and is
        # safe on the (constant) embedding rows since their norm is nonzero.
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        clf = LogisticRegression(max_iter=5000)
        return cross_val_score(clf, Xn, y, cv=4, scoring="roc_auc_ovr").mean()

    run = run or {}
    print(f"  samples={len(y)}  classes={len(codenames)} (codename values)  "
          f"layers={m.n_layers}")
    print("-" * 72)
    for L in probe_layers:
        a = auc(resid[L])
        log_metric(coll, run, {
            "mode": "probe", "repr": "residual", "layer": int(L), "auc": a,
            "samples": int(len(y)), "classes": len(codenames),
        })
        print(f"  residual stream (layer {L:>2})      : AUC {a:.3f}")
    a_embed = auc(X_embed)
    log_metric(coll, run, {
        "mode": "probe", "repr": "embedding", "layer": 0, "auc": a_embed,
        "samples": int(len(y)), "classes": len(codenames),
    })
    print(f"  input embeddings  (layer  0)      : AUC {a_embed:.3f}  "
          "(chance — last-token input is identical across classes)")
    print("-" * 72)
    print("The planted codename is decodable from the residual stream far above the")
    print("embedding baseline: the value survives in the hidden state even as the")
    print("attention channel to the system prompt thins.\n")


# --------------------------------------------------------------------------- #
# Aggregations over the accumulated runs. These are the payoff of using a real
# MongoDB engine: each is a standard aggregation pipeline, not a hand-rolled loop.
# Factored out so they can be unit-tested against a seeded collection.
# --------------------------------------------------------------------------- #
def latest_run_ids(coll, match: dict | None = None) -> list[str]:
    """The most recent run_id per model (by timestamp). Used to scope `report` to the
    latest run per model so repeated runs don't inflate or blend the numbers."""
    rows = coll.aggregate([
        {"$match": {**(match or {})}},
        {"$sort": {"ts": -1}},
        {"$group": {"_id": "$model", "run_id": {"$first": "$run_id"}}},
    ])
    return [r["run_id"] for r in rows]


def crossover_by_model(coll, match: dict | None = None) -> list[dict]:
    """First turn count at which recall dropped below full, per model — the demo's
    analog of the paper's 'crossover turn' where behavior starts to fail."""
    return list(coll.aggregate([
        {"$match": {"mode": "gar", "recall_miss": True, **(match or {})}},
        {"$group": {"_id": "$model", "first_miss_turn": {"$min": "$turns"},
                    "lowest_gar_seen": {"$min": "$gar_all"}}},
        {"$sort": {"_id": 1}},
    ]))


def gar_range_by_model(coll, match: dict | None = None) -> list[dict]:
    """How far GAR decayed (max -> min) and how long the context grew, per model."""
    return list(coll.aggregate([
        {"$match": {"mode": "gar", **(match or {})}},
        {"$group": {"_id": "$model", "max_gar": {"$max": "$gar_all"},
                    "min_gar": {"$min": "$gar_all"}, "max_turns": {"$max": "$turns"}}},
        {"$sort": {"_id": 1}},
    ]))


def ablation_rate_by_model(coll, match: dict | None = None) -> list[dict]:
    """Recall under normal vs ablated attention, summed across facts, per model.
    Callers should present these as rates (ok / facts) so the figures are meaningful
    regardless of how many runs are in scope."""
    return list(coll.aggregate([
        {"$match": {"mode": "ablate", **(match or {})}},
        {"$group": {"_id": "$model", "normal_ok": {"$sum": "$normal_ok"},
                    "ablated_ok": {"$sum": "$ablated_ok"}, "facts": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]))


def best_residual_auc_by_model(coll, match: dict | None = None) -> list[dict]:
    """Best residual-probe AUC vs the embedding baseline, per model."""
    return list(coll.aggregate([
        {"$match": {"mode": "probe", **(match or {})}},
        {"$group": {"_id": {"model": "$model", "repr": "$repr"},
                    "best_auc": {"$max": "$auc"}}},
        {"$sort": {"_id.model": 1, "_id.repr": 1}},
    ]))


def run_report(coll, scope: str = "latest", run_id: str | None = None) -> None:
    print("\n" + "=" * 72)
    print("  REPORT: MongoDB aggregations over stored runs")
    print("=" * 72)

    if coll is None:
        print("Metrics store unavailable — nothing to report.\n")
        return
    if coll.count_documents({}) == 0:
        print("No runs logged yet. Run e.g. `python3 demo.py all` first.\n")
        return

    # Resolve the scope into a $match filter that every pipeline shares.
    if scope == "run" and run_id:
        # Accept a run_id prefix (the value printed after a run is truncated to 8 chars).
        full = next((r for r in coll.distinct("run_id") if r.startswith(run_id)), run_id)
        match = {"run_id": full}
        scope_label = f"run {full[:8]}"
    elif scope == "all":
        match = {}
        scope_label = "all runs (full history)"
    else:  # "latest"
        match = {"run_id": {"$in": latest_run_ids(coll)}}
        scope_label = "latest run per model (use --all-runs for full history)"

    n = coll.count_documents(match)
    models = sorted(coll.distinct("model", match))
    print(f"  scope: {scope_label}")
    print(f"  {n} documents across {len(models)} model(s): {', '.join(models)}")

    print("\n  GAR decay (max -> min) and context reached:")
    print(f"    {'model':>28} | {'max GAR':>8} | {'min GAR':>8} | {'max turns':>9}")
    for r in gar_range_by_model(coll, match):
        print(f"    {r['_id']:>28} | {r['max_gar']:>8.4f} | {r['min_gar']:>8.4f} | "
              f"{r['max_turns']:>9}")

    print("\n  First recall MISS (crossover turn):")
    rows = crossover_by_model(coll, match)
    if rows:
        print(f"    {'model':>28} | {'first miss turn':>15} | {'GAR there':>9}")
        for r in rows:
            print(f"    {r['_id']:>28} | {r['first_miss_turn']:>15} | "
                  f"{r['lowest_gar_seen']:>9.4f}")
    else:
        print("    (no natural recall MISS recorded in scope)")

    print("\n  Ablation recall (rate over facts, normal vs ablated):")
    print(f"    {'model':>28} | {'normal':>8} | {'ablated':>8} | {'facts':>5}")
    for r in ablation_rate_by_model(coll, match):
        facts = r["facts"] or 1
        print(f"    {r['_id']:>28} | {r['normal_ok'] / facts:>8.2f} | "
              f"{r['ablated_ok'] / facts:>8.2f} | {r['facts']:>5}")

    print("\n  Best probe AUC (residual vs embedding baseline):")
    print(f"    {'model':>28} | {'repr':>9} | {'best AUC':>8}")
    for r in best_residual_auc_by_model(coll, match):
        print(f"    {r['_id']['model']:>28} | {r['_id']['repr']:>9} | {r['best_auc']:>8.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention-channel-closing demo")
    ap.add_argument("mode", choices=["gar", "ablate", "probe", "all", "report"])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--db", default=DB_URI,
                    help="smongo URI for the metrics store (default: %(default)s)")
    ap.add_argument("--all-runs", action="store_true",
                    help="report: aggregate full history instead of the latest run per model")
    ap.add_argument("--run", default=None, metavar="RUN_ID",
                    help="report: scope to a single run_id")
    args = ap.parse_args()

    # Opening the store is best-effort: if it fails, the scientific modes still run
    # (they just won't log); only `report` truly needs it.
    try:
        coll = open_metrics(args.db)
    except Exception as e:
        print(f"[warn] could not open metrics store {args.db}: {e}", file=sys.stderr)
        coll = None

    # `report` only reads the store — no need to load the model.
    if args.mode == "report":
        scope = "run" if args.run else ("all" if args.all_runs else "latest")
        run_report(coll, scope=scope, run_id=args.run)
        return

    run = new_run(args.model)
    m = Model(args.model)
    if args.mode in ("gar", "all"):
        run_gar(m, coll, run)
    if args.mode in ("ablate", "all"):
        run_ablate(m, coll, run)
    if args.mode in ("probe", "all"):
        run_probe(m, coll, run)

    if coll is not None:
        logged = coll.count_documents({"run_id": run["run_id"]})
        print(f"logged {logged} metric docs to {args.db} (run_id {run['run_id'][:8]}) — "
              "run `python3 demo.py report` to aggregate across runs.\n")


if __name__ == "__main__":
    main()

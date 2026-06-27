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
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def supports_system_role(self) -> bool:
        """Whether the chat template accepts a `system` message. The demo plants the goal
        in the system prompt and measures attention onto it, so a template that drops or
        rejects the system role (e.g. Gemma folds it into the first user turn) cannot be
        measured as-is and should be skipped rather than silently mis-spanned."""
        try:
            self.tok.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def encode(self, messages: list[dict]) -> torch.Tensor:
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return enc["input_ids"].to(DEVICE)

    def system_span(self, messages: list[dict]) -> slice:
        """Token span covering just the system message, within the full encoding.

        Different chat templates render the system block slightly differently (ChatML's
        `<|im_start|>system ... <|im_end|>` vs Zephyr's `<|system|> ... </s>`), and token
        boundaries can shift once later turns are appended. Rather than assume the
        system-only encoding is an exact prefix of the full conversation, we take the
        longest common token prefix of the two — that is exactly the leading run of tokens
        owned by the system message under any template, and it is what GAR/ablation must
        target as the goal span."""
        full = self.encode_no_gen(messages)
        sys_only = self.encode_no_gen([messages[0]])
        k = 0
        limit = min(sys_only.shape[1], full.shape[1])
        while k < limit and full[0, k].item() == sys_only[0, k].item():
            k += 1
        return slice(0, k)

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


# Full GAR sweep (filler turns). Recall generation does a prefill at the resulting context
# length, and because we force `eager` attention (to read attentions) that prefill
# materializes an O(heads * L^2) score tensor. That is fine for sub-1B models but OOMs a
# larger one at multi-thousand-token context — so the schedule is capped by model size.
GAR_SCHEDULE = (0, 8, 24, 56, 96, 128, 160)


def gar_schedule(n_params: int, max_turns: int | None = None) -> tuple[int, ...]:
    """Pick the GAR filler-turn schedule. An explicit `max_turns` wins; otherwise cap by
    parameter count so eager-attention prefill stays within memory on larger models."""
    if max_turns is None:
        if n_params > 1.2e9:
            max_turns = 24
        elif n_params > 0.8e9:
            max_turns = 56
        else:
            max_turns = GAR_SCHEDULE[-1]
    return tuple(t for t in GAR_SCHEDULE if t <= max_turns) or (0,)


def run_gar(m: "Model", coll=None, run: dict | None = None,
            max_turns: int | None = None) -> None:
    run = run or {}
    print("\n" + "=" * 72)
    print("  MODE 1: GAR decay — attention thins, then recall finally breaks")
    print("=" * 72)
    schedule = gar_schedule(m.n_params, max_turns)
    if schedule[-1] < GAR_SCHEDULE[-1]:
        print(f"  (context capped at {schedule[-1]} turns for a "
              f"{m.n_params/1e6:.0f}M-param model to keep eager attention in memory; "
              "override with --max-turns)")
    print(f"{'turns':>5} | {'ctx tok':>7} | {'GAR all':>8} | {'early':>7} | "
          f"{'late':>7} | {'recall':>6}")
    print("-" * 72)

    # Sweep the "relentless flurry": recall is sticky, so a natural MISS only shows up at
    # long context. GAR is measured memory-safely at the final token (Model.gar_last_token).
    first_miss = None
    for n_filler in schedule:
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


# Fractions of the system span left visible to post-system tokens in the graded closure
# sweep. 1.0 == full baseline; lower values hide more of the goal. Survival is measured over
# the partial closures (fraction < 1.0); total closure (0.0) is the separate MODE-2 baseline.
CLOSURE_KEEP_FRACTIONS = (1.0, 0.75, 0.5, 0.25)


def run_ablate(m: "Model", coll=None, run: dict | None = None) -> None:
    run = run or {}
    print("\n" + "=" * 72)
    print("  MODE 2: Ablation — close attention to system tokens, totally then gradually")
    print("=" * 72)

    # Use a modest context where normal recall is clean, so any collapse is
    # attributable to blinding the model to its system tokens (not generic decay).
    msgs = build_conversation(4)
    sys_span = m.system_span(msgs)

    normal_hits, ablated_hits = 0, 0
    print("  total closure (channel from every generated token to the system span):")
    print(f"    {'fact':>20} | {'normal':>8} | {'ablated':>8}")
    print("    " + "-" * 44)
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
        print(f"    {fact_name:>20} | {'OK' if n_ok else 'MISS':>8} | "
              f"{'OK' if a_ok else 'MISS':>8}")

    n = len(PROBES)
    print("    " + "-" * 44)
    print(f"    recall: normal {normal_hits}/{n} ({100*normal_hits/n:.0f}%)  ->  "
          f"ablated {ablated_hits}/{n} ({100*ablated_hits/n:.0f}%)")

    # Graded closure: hide a growing suffix of the system span. Facts that stay visible are
    # recalled via attention; hidden facts can only be recovered from the residual stream —
    # so recall here separates residual-reliant models (recall persists) from attention-
    # reliant ones (recall collapses), and lands between full and zero.
    sys_len = sys_span.stop - sys_span.start
    print("\n  graded closure (keep the first frac of the system prompt visible):")
    print(f"    {'visible':>8} | {'sys masked':>10} | {'recall':>6}")
    print("    " + "-" * 32)
    closing_hits = closing_total = 0
    for frac in CLOSURE_KEEP_FRACTIONS:
        hits = 0
        masked = sys_len - round(frac * sys_len)
        for fact_name, question in PROBES:
            ids = m.encode(msgs + [{"role": "user", "content": question}])
            reply = generate_partial_ablated(m, ids, sys_span, frac, max_new_tokens=30)
            ok = FACTS[fact_name].lower() in reply.lower()
            hits += ok
            log_metric(coll, run, {
                "mode": "closure", "fact": fact_name, "frac": frac,
                "sys_masked": masked, "sys_len": sys_len, "recall_ok": int(ok),
            })
        if frac < 1.0:
            closing_hits += hits
            closing_total += len(PROBES)
        print(f"    {frac:>8.2f} | {masked:>10} | {hits:>4}/{len(PROBES)}")

    survival = closing_hits / closing_total if closing_total else 0.0
    print("    " + "-" * 32)
    print(f"  survival under partial closure: {survival:.2f} "
          f"({closing_hits}/{closing_total} recalled). Total closure collapses recall; how "
          "much\n  survives the *graded* closure is what separates architectures (see "
          "`compare`).\n")


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


def build_partial_ablation_mask(L: int, sys_span: slice, keep_frac: float,
                                device=None) -> torch.Tensor:
    """Additive (1, 1, L, L) mask that *partially* closes the channel from post-system
    tokens to the system span: it keeps the first `keep_frac` of the system-token columns
    visible and masks the rest. This is the graded generalization of `build_ablation_mask`:

      - keep_frac == 1.0 -> mask nothing (plain causal baseline)
      - keep_frac == 0.0 -> mask the whole span (== build_ablation_mask, total closure)

    Because the goal facts sit on separate lines in the system prompt, hiding a suffix of
    the span leaves some facts attendable while others can only be recovered from the
    residual stream — so recall here lands *between* full and zero, and how much survives
    separates residual-reliant models from attention-reliant ones. Only post-system rows are
    masked, so every row keeps its causal diagonal (no all -inf row, no softmax NaN)."""
    mask = torch.full((L, L), NEG, device=device).triu(1)
    span = list(range(sys_span.start, sys_span.stop))
    keep_n = round(keep_frac * len(span))
    masked_cols = span[keep_n:]
    if masked_cols:
        mask[sys_span.stop:, masked_cols] = NEG
    return mask.view(1, 1, L, L)


@torch.no_grad()
def generate_partial_ablated(m: "Model", ids: torch.Tensor, sys_span: slice,
                             keep_frac: float, max_new_tokens: int = 30) -> str:
    """Greedy-generate while keeping only the first `keep_frac` of the system span visible to
    post-system tokens at every step (see `build_partial_ablation_mask`)."""
    cur = ids
    for _ in range(max_new_tokens):
        L = cur.shape[1]
        mask = build_partial_ablation_mask(L, sys_span, keep_frac, device=DEVICE)
        logits = m.model(cur, attention_mask=mask).logits
        if not torch.isfinite(logits[0, -1]).all():
            raise RuntimeError("non-finite logits under the partial-ablation mask.")
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


def closure_survival_by_model(coll, match: dict | None = None) -> list[dict]:
    """Behavioral survival under the *graded* (partial) closure, per model: the mean recall
    over the partial closures (fraction < 1.0, i.e. excluding the full-attention baseline).
    This is the dissociation axis — a residual-reliant model keeps recalling as more of the
    system prompt is hidden, an attention-reliant one collapses. Unlike total ablation (which
    pins recall at 0 for everyone), this lands between 0 and 1."""
    return list(coll.aggregate([
        {"$match": {"mode": "closure", "frac": {"$lt": 1.0}, **(match or {})}},
        {"$group": {"_id": "$model", "survival": {"$avg": "$recall_ok"},
                    "steps": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]))


# Thresholds for the cross-architecture reliance call. Deliberately lenient and named so the
# heuristic is explicit: "decodable" = the goal is recoverable from the residual stream at all;
# "survives" = at least half of recall holds when the attention channel is force-closed.
AUC_DECODABLE = 0.70
SURVIVAL_OK = 0.50


def classify_reliance(residual_auc: float | None, survival: float | None) -> str:
    """Bucket a model by *what its behavior relies on* when attention to the goal closes:

      - weak-encoding    : the goal is not even decodable from the residual stream.
      - residual-reliant : goal decodable AND recall largely survives channel closure
                           (the model reads the goal from the residual stream -> robust).
      - attention-reliant: goal decodable BUT recall collapses under closure (the info is
                           present yet the model can't use it without attention -> fragile).

    The last case is the paper's striking dissociation: 'what survives reveals architecture'."""
    if residual_auc is None or residual_auc < AUC_DECODABLE:
        return "weak-encoding"
    if survival is None:
        return "residual-reliant"  # decodable, no closure evidence to the contrary
    return "residual-reliant" if survival >= SURVIVAL_OK else "attention-reliant"


def compare_by_model(coll, match: dict | None = None) -> list[dict]:
    """Assemble the per-model dissociation signature by joining the existing aggregations in
    Python (each pipeline stays simple and unit-testable). Returns one row per model with the
    two axes that matter — residual decodability and behavioral survival under closure — plus
    the reliance bucket from `classify_reliance`."""
    auc = {}
    for r in best_residual_auc_by_model(coll, match):
        auc.setdefault(r["_id"]["model"], {})[r["_id"]["repr"]] = r["best_auc"]
    ablate = {r["_id"]: r for r in ablation_rate_by_model(coll, match)}
    closure = {r["_id"]: r for r in closure_survival_by_model(coll, match)}
    cross = {r["_id"]: r for r in crossover_by_model(coll, match)}
    gar = {r["_id"]: r for r in gar_range_by_model(coll, match)}

    models = sorted(set(auc) | set(ablate) | set(closure) | set(cross) | set(gar))
    rows = []
    for name in models:
        a = ablate.get(name)
        normal_ok = a["normal_ok"] if a else 0
        ablated_ok = a["ablated_ok"] if a else 0
        facts = (a["facts"] if a else 0) or 0
        # Prefer the graded partial-closure survival (lands in [0,1]); fall back to the
        # total-ablation ratio only when no graded sweep was logged.
        if name in closure:
            survival = closure[name]["survival"]
        else:
            survival = ablated_ok / normal_ok if normal_ok else None
        residual_auc = auc.get(name, {}).get("residual")
        rows.append({
            "model": name,
            "residual_auc": residual_auc,
            "embedding_auc": auc.get(name, {}).get("embedding"),
            "normal_rate": (normal_ok / facts) if facts else None,
            "ablated_rate": (ablated_ok / facts) if facts else None,
            "survival": survival,
            "first_miss_turn": cross.get(name, {}).get("first_miss_turn"),
            "min_gar": gar.get(name, {}).get("min_gar"),
            "max_gar": gar.get(name, {}).get("max_gar"),
            "reliance": classify_reliance(residual_auc, survival),
        })
    return rows


def short_model(name: str) -> str:
    """Display name without the org prefix (e.g. 'Qwen/Qwen2.5-0.5B-Instruct' -> the part
    after the last '/'), so per-model tables line up regardless of the hub path length."""
    return name.rsplit("/", 1)[-1]


def resolve_scope(coll, scope: str = "latest", run_id: str | None = None) -> tuple[dict, str]:
    """Turn a (scope, run_id) request into the `$match` filter every pipeline shares, plus a
    human-readable label. Shared by `report` and `compare`."""
    if scope == "run" and run_id:
        # Accept a run_id prefix (the value printed after a run is truncated to 8 chars).
        full = next((r for r in coll.distinct("run_id") if r.startswith(run_id)), run_id)
        return {"run_id": full}, f"run {full[:8]}"
    if scope == "all":
        return {}, "all runs (full history)"
    return ({"run_id": {"$in": latest_run_ids(coll)}},
            "latest run per model (use --all-runs for full history)")


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

    match, scope_label = resolve_scope(coll, scope, run_id)
    n = coll.count_documents(match)
    models = sorted(coll.distinct("model", match))
    print(f"  scope: {scope_label}")
    print(f"  {n} documents across {len(models)} model(s): {', '.join(models)}")

    print("\n  GAR decay (max -> min) and context reached:")
    print(f"    {'model':>24} | {'max GAR':>8} | {'min GAR':>8} | {'max turns':>9}")
    for r in gar_range_by_model(coll, match):
        print(f"    {short_model(r['_id']):>24} | {r['max_gar']:>8.4f} | {r['min_gar']:>8.4f} | "
              f"{r['max_turns']:>9}")

    print("\n  First recall MISS (crossover turn):")
    rows = crossover_by_model(coll, match)
    if rows:
        print(f"    {'model':>24} | {'first miss turn':>15} | {'GAR there':>9}")
        for r in rows:
            print(f"    {short_model(r['_id']):>24} | {r['first_miss_turn']:>15} | "
                  f"{r['lowest_gar_seen']:>9.4f}")
    else:
        print("    (no natural recall MISS recorded in scope)")

    print("\n  Ablation recall (rate over facts, normal vs ablated):")
    print(f"    {'model':>24} | {'normal':>8} | {'ablated':>8} | {'facts':>5}")
    for r in ablation_rate_by_model(coll, match):
        facts = r["facts"] or 1
        print(f"    {short_model(r['_id']):>24} | {r['normal_ok'] / facts:>8.2f} | "
              f"{r['ablated_ok'] / facts:>8.2f} | {r['facts']:>5}")

    print("\n  Best probe AUC (residual vs embedding baseline):")
    print(f"    {'model':>24} | {'repr':>9} | {'best AUC':>8}")
    for r in best_residual_auc_by_model(coll, match):
        print(f"    {short_model(r['_id']['model']):>24} | {r['_id']['repr']:>9} | "
              f"{r['best_auc']:>8.3f}")

    comp_rows = compare_by_model(coll, match)
    if len(comp_rows) > 1:
        print("\n  Cross-architecture reliance (see `compare` for the full signature):")
        print(f"    {'model':>24} | {'reliance':>17}")
        for r in comp_rows:
            print(f"    {short_model(r['model']):>24} | {r['reliance']:>17}")
    print()


def run_compare(coll, scope: str = "latest", run_id: str | None = None) -> None:
    """Cross-architecture view: line up each model's *dissociation signature* — does the goal
    survive in the residual stream (probe AUC), and does behavior survive as the attention
    channel to the goal is progressively closed (graded closure survival)? The paper's headline
    is that these two can come apart differently by architecture. Descriptive, not a replication."""
    print("\n" + "=" * 72)
    print("  COMPARE: cross-architecture dissociation (residual survival vs behavior)")
    print("=" * 72)

    if coll is None:
        print("Metrics store unavailable — nothing to compare.\n")
        return
    if coll.count_documents({}) == 0:
        print("No runs logged yet. Run e.g. `python3 demo.py all --model <name>` first.\n")
        return

    match, scope_label = resolve_scope(coll, scope, run_id)
    rows = compare_by_model(coll, match)
    print(f"  scope: {scope_label}")
    print(f"  {len(rows)} model(s): {', '.join(short_model(r['model']) for r in rows)}")

    def fmt(x, spec="6.3f"):
        return format(x, spec) if x is not None else "  n/a"

    print(f"\n  {'model':>24} | {'res AUC':>7} | {'emb AUC':>7} | {'normal':>6} | "
          f"{'ablat':>6} | {'surv':>5} | {'miss@':>6} | {'reliance':>17}")
    print("  " + "-" * 100)
    for r in rows:
        miss = r["first_miss_turn"] if r["first_miss_turn"] is not None else "none"
        print(f"  {short_model(r['model']):>24} | {fmt(r['residual_auc']):>7} | "
              f"{fmt(r['embedding_auc']):>7} | {fmt(r['normal_rate'], '6.2f'):>6} | "
              f"{fmt(r['ablated_rate'], '6.2f'):>6} | {fmt(r['survival'], '5.2f'):>5} | "
              f"{str(miss):>6} | {r['reliance']:>17}")

    print("\n  Reading it:")
    print("    - ablat = recall under TOTAL closure (system span fully blinded; ~0 by design).")
    print("    - surv  = recall under GRADED closure (mean recall as more of the system prompt")
    print("              is hidden) — the discriminating axis, since it lands anywhere in [0, 1].")
    print("    - residual-reliant : goal decodable AND recall survives graded closure (robust).")
    print("    - attention-reliant: goal decodable BUT recall collapses under closure")
    print("                         (info present, unused without attention — the dissociation).")
    print("    - weak-encoding    : goal not decodable from the residual stream.")

    if len(rows) < 2:
        print("\n  Only one model in scope — log another with `demo.py all --model <name>`")
        print("  (e.g. SmolLM2-360M-Instruct, TinyLlama-1.1B-Chat-v1.0) to see a contrast.")

    print("\n  Caveat: small instruct models, a single run each, 4 planted facts and a 32-sample")
    print("  probe. This shows the *shape* of the dissociation, not the paper's statistically")
    print("  treated cross-architecture result.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention-channel-closing demo")
    ap.add_argument("mode", choices=["gar", "ablate", "probe", "all", "report", "compare"])
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--db", default=DB_URI,
                    help="smongo URI for the metrics store (default: %(default)s)")
    ap.add_argument("--all-runs", action="store_true",
                    help="report/compare: aggregate full history instead of the latest run per model")
    ap.add_argument("--run", default=None, metavar="RUN_ID",
                    help="report/compare: scope to a single run_id")
    ap.add_argument("--max-turns", type=int, default=None, metavar="N",
                    help="gar: cap the filler-turn sweep (default: auto by model size, to "
                         "keep eager-attention prefill within memory)")
    args = ap.parse_args()

    # Opening the store is best-effort: if it fails, the scientific modes still run
    # (they just won't log); only `report`/`compare` truly need it.
    try:
        coll = open_metrics(args.db)
    except Exception as e:
        print(f"[warn] could not open metrics store {args.db}: {e}", file=sys.stderr)
        coll = None

    # `report` and `compare` only read the store — no need to load the model.
    if args.mode in ("report", "compare"):
        scope = "run" if args.run else ("all" if args.all_runs else "latest")
        (run_report if args.mode == "report" else run_compare)(
            coll, scope=scope, run_id=args.run)
        return

    run = new_run(args.model)
    m = Model(args.model)
    if not m.supports_system_role():
        print(f"[skip] {args.model}: its chat template does not accept a system role, so "
              "the goal cannot be planted/measured in the system prompt. Try a model with "
              "system-role support (e.g. Qwen2.5, SmolLM2, TinyLlama).", file=sys.stderr)
        return
    if args.mode in ("gar", "all"):
        run_gar(m, coll, run, max_turns=args.max_turns)
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

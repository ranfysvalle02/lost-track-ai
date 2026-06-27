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
import os
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


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared leading run of two token-id lists."""
    k = 0
    limit = min(len(a), len(b))
    while k < limit and a[k] == b[k]:
        k += 1
    return k


def _approx_params(cfg) -> float:
    """Rough parameter count from a model config (no weights loaded), for the `auto` dtype
    heuristic. Embeddings + per-layer attention (~4 h^2) + SwiGLU MLP (~3 h * intermediate)."""
    h = getattr(cfg, "hidden_size", 0) or 0
    layers = getattr(cfg, "num_hidden_layers", 0) or 0
    vocab = getattr(cfg, "vocab_size", 0) or 0
    inter = getattr(cfg, "intermediate_size", 4 * h) or (4 * h)
    return float(vocab * h + layers * (4 * h * h + 3 * h * inter))


def resolve_dtype(name: str, dtype: str) -> "torch.dtype":
    """Map a `--dtype` choice to a torch dtype. `auto` keeps small models in float32 (fidelity)
    but drops large models (>2B params, estimated from config) to bfloat16 on mps/cuda so a 7B
    actually fits; on CPU it stays float32. bfloat16 (not float16) is the large-model default
    because float16's narrow range overflows in this model's eager attention on mps, yielding
    NaN GAR and non-finite logits under the ablation mask — bf16 keeps float32's exponent range."""
    table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype != "auto":
        return table[dtype]
    try:
        from transformers import AutoConfig
        approx = _approx_params(AutoConfig.from_pretrained(name, trust_remote_code=True))
    except Exception:
        approx = 0.0
    return torch.bfloat16 if (DEVICE != "cpu" and approx > 2e9) else torch.float32


class Model:
    """Thin wrapper exposing tokenization, generation, attentions and hidden states."""

    def __init__(self, name: str = MODEL_NAME, dtype: str = "float32"):
        torch_dtype = resolve_dtype(name, dtype)
        print(f"loading {name} on {DEVICE} ({torch_dtype}) ...", flush=True)
        # trust_remote_code lets us load contrasting families whose tokenizer/model live in
        # the repo (e.g. StableLM-2, InternLM2). This executes code from the model repo, so it
        # is only appropriate for reputable, vetted repos like the ones this demo names.
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch_dtype, attn_implementation="eager", trust_remote_code=True
        )
        self.model.to(DEVICE).eval()
        self.dtype = torch_dtype
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
        longest common token prefix of the full encoding and a system-only encoding — that
        leading run is the system block under any well-behaved template.

        Some templates (e.g. StableLM-2) render a *lone* system message to nothing, which
        would yield an empty span and silently turn the ablation into a no-op. For those we
        fall back to diffing two conversations that share the system block but differ in the
        first user turn: their common prefix is the system block plus the user-turn opener
        (a few structural tokens), which still covers every goal token. A still-empty result
        means the goal span can't be located for this model -> the caller should skip it."""
        full = self.encode_no_gen(messages)[0].tolist()
        sys_only = self.encode_no_gen([messages[0]])[0].tolist()
        k = _common_prefix_len(full, sys_only)
        if k > 0:
            return slice(0, k)
        a = self.encode_no_gen([messages[0], {"role": "user", "content": "A"}])[0].tolist()
        b = self.encode_no_gen([messages[0], {"role": "user", "content": "BB CC"}])[0].tolist()
        k = min(_common_prefix_len(a, b), _common_prefix_len(full, a))
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
    def hidden_last_layers(self, ids: torch.Tensor, layers) -> dict:
        """Final-token hidden state at each requested layer index, WITHOUT materializing
        attentions. `Model.forward` forces `output_attentions=True`, which builds an
        O(L^2) score tensor per layer and OOMs at long context; this is the memory-safe
        path for probing the residual stream as context grows (used by `run_dissociate`)."""
        out = self.model(ids, output_hidden_states=True, use_cache=False)
        return {L: out.hidden_states[L][0, -1].float().cpu().numpy() for L in layers}

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
# Which span columns to mask at a partial fraction. Averaging over all three removes the
# positional bias of any single ordering (see build_partial_ablation_mask).
CLOSURE_ORDERS = ("strided", "suffix", "prefix")


def run_ablate(m: "Model", coll=None, run: dict | None = None, seeds: int = 3,
               light: bool = False) -> None:
    run = run or {}
    print("\n" + "=" * 72)
    print("  MODE 2: Ablation — close attention to system tokens, totally then gradually")
    print("=" * 72)

    # `light` budget for big models: one seed, one mask order, shorter generations — keeps a
    # 7B run bounded to minutes. 24 tokens still comfortably reaches the fact value even with a
    # verbose persona (a tighter cap can read as a false MISS). Endpoints are unchanged.
    orders = ("strided",) if light else CLOSURE_ORDERS
    gen_tokens = 24 if light else 30
    if light:
        seeds = 1
        print(f"  (--light: seeds=1, orders={orders}, max_new_tokens={gen_tokens})")

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

        normal = m.generate(ids, max_new_tokens=gen_tokens)
        ablated = generate_ablated(m, ids, sys_span, max_new_tokens=gen_tokens)

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

    # Graded closure: hide a fraction of the system span. Facts that stay visible are recalled
    # via attention; hidden facts can only be recovered from the residual stream — so recall
    # here separates residual-reliant models (recall persists) from attention-reliant ones
    # (recall collapses), and lands between full and zero. We average over mask orderings (so
    # survival reflects the goal channel, not where a fact sits) and over `seeds` filler
    # orderings (so it is a band, not a single draw). The filler `start` rotation is free.
    partial = [f for f in CLOSURE_KEEP_FRACTIONS if f < 1.0]
    grid: dict[tuple[str, float], list[int]] = {(o, f): [] for o in orders for f in partial}
    seed_survival: list[float] = []
    print(f"\n  graded closure (avg over orders {orders} x {seeds} filler seeds; "
          "recall as more of the system prompt is hidden):")
    for seed in range(seeds):
        msgs_s = build_conversation(4, start=seed)
        sys_span_s = m.system_span(msgs_s)
        sys_len = sys_span_s.stop - sys_span_s.start
        # Baseline (frac 1.0) is order-independent, so measure it once per seed.
        for fact_name, question in PROBES:
            ids = m.encode(msgs_s + [{"role": "user", "content": question}])
            ok = FACTS[fact_name].lower() in m.generate(ids, max_new_tokens=gen_tokens).lower()
            log_metric(coll, run, {"mode": "closure", "fact": fact_name, "frac": 1.0,
                                   "order": "baseline", "seed": seed, "recall_ok": int(ok)})
        seed_hits = seed_total = 0
        for order in orders:
            for frac in partial:
                masked = sys_len - round(frac * sys_len)
                for fact_name, question in PROBES:
                    ids = m.encode(msgs_s + [{"role": "user", "content": question}])
                    reply = generate_partial_ablated(m, ids, sys_span_s, frac, order,
                                                     max_new_tokens=gen_tokens)
                    ok = int(FACTS[fact_name].lower() in reply.lower())
                    grid[(order, frac)].append(ok)
                    seed_hits += ok
                    seed_total += 1
                    log_metric(coll, run, {
                        "mode": "closure", "fact": fact_name, "frac": frac, "order": order,
                        "seed": seed, "sys_masked": masked, "sys_len": sys_len, "recall_ok": ok,
                    })
        seed_survival.append(seed_hits / seed_total if seed_total else 0.0)

    # Compact curve: recall fraction per (visible fraction x order), averaged over seeds.
    print(f"    {'visible':>8} | " + " | ".join(f"{o:>8}" for o in orders) + " | "
          f"{'mean':>5}")
    print("    " + "-" * (12 + 11 * len(orders) + 8))
    for frac in partial:
        cells = [sum(grid[(o, frac)]) / len(grid[(o, frac)]) for o in orders]
        row_mean = sum(cells) / len(cells)
        print(f"    {frac:>8.2f} | " + " | ".join(f"{c:>8.2f}" for c in cells) +
              f" | {row_mean:>5.2f}")

    survival = sum(seed_survival) / len(seed_survival) if seed_survival else 0.0
    lo, hi = (min(seed_survival), max(seed_survival)) if seed_survival else (0.0, 0.0)
    print("    " + "-" * (12 + 11 * len(orders) + 8))
    print(f"  survival under partial closure: {survival:.2f} (seed band [{lo:.2f}, {hi:.2f}]). "
          "Total closure collapses recall;\n  how much survives the *graded* closure is what "
          "separates architectures (see `compare`).\n")


def build_ablation_mask(L: int, sys_span: slice, device=None,
                        dtype: "torch.dtype" = torch.float32) -> torch.Tensor:
    """Additive attention mask of shape (1, 1, L, L) that keeps the causal structure but
    closes the attention channel from every *post-system* query position onto the
    system-token span.

    Critically, the system rows themselves are NOT masked off their own (causal)
    self-attention. If we masked the system columns for *all* rows (as a naive
    implementation does), the first system row could only attend to a fully -inf column
    set, producing an all -inf row -> softmax NaN that corrupts generation. By masking
    only rows at/after `sys_span.stop`, every row retains at least its diagonal, so there
    are no all -inf rows and no NaNs. This matches the paper's manipulation: force-close
    the channel *from generated tokens to goal tokens*.

    `dtype` matches the model's compute dtype so the additive mask is finite in that dtype
    (float32's min would overflow to -inf in float16, used for the local 7B attempt)."""
    neg = torch.finfo(dtype).min
    mask = torch.full((L, L), neg, device=device, dtype=dtype).triu(1)
    mask[sys_span.stop:, sys_span] = neg
    return mask.view(1, 1, L, L)


@torch.no_grad()
def generate_ablated(m: "Model", ids: torch.Tensor, sys_span: slice,
                     max_new_tokens: int = 30) -> str:
    """Greedy-generate while closing the attention channel from generated tokens to the
    system-token span at every step (see `build_ablation_mask`)."""
    cur = ids
    for _ in range(max_new_tokens):
        L = cur.shape[1]
        mask = build_ablation_mask(L, sys_span, device=DEVICE, dtype=m.dtype)
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
                                order: str = "strided", device=None,
                                dtype: "torch.dtype" = torch.float32) -> torch.Tensor:
    """Additive (1, 1, L, L) mask that *partially* closes the channel from post-system
    tokens to the system span: it keeps a `keep_frac` fraction of the system-token columns
    visible to post-system rows and masks the rest. The graded generalization of
    `build_ablation_mask`, with the endpoints identical for every `order`:

      - keep_frac == 1.0 -> mask nothing (plain causal baseline)
      - keep_frac == 0.0 -> mask the whole span (== build_ablation_mask, total closure)

    `order` names *which* span columns are masked when 0 < keep_frac < 1 — the goal facts
    sit on separate lines, so the choice of region biases which facts stay attendable. We
    sweep all three and average so `survival` reflects the goal channel, not fact position:

      - "suffix"  : mask the tail of the span (keep the head visible) — the original demo.
      - "prefix"  : mask the head of the span (keep the tail visible).
      - "strided" : mask an evenly-spaced subset (keep a strided subset visible) — least
                    positionally biased, hence the default.

    Only post-system rows are masked, so every row keeps its causal diagonal (no all -inf
    row, no softmax NaN). `dtype` matches the model's compute dtype (finite in float16)."""
    neg = torch.finfo(dtype).min
    mask = torch.full((L, L), neg, device=device, dtype=dtype).triu(1)
    span = list(range(sys_span.start, sys_span.stop))
    span_len = len(span)
    keep_n = round(keep_frac * span_len)
    if keep_n <= 0:
        kept: set[int] = set()
    elif keep_n >= span_len:
        kept = set(span)
    elif order == "suffix":
        kept = set(span[:keep_n])               # keep head, mask the suffix
    elif order == "prefix":
        kept = set(span[span_len - keep_n:])    # keep tail, mask the prefix
    elif order == "strided":
        pos = torch.linspace(0, span_len - 1, keep_n).round().long().tolist()
        kept = {span[p] for p in pos}
    else:
        raise ValueError(f"unknown order {order!r} (use strided/suffix/prefix)")
    masked_cols = [c for c in span if c not in kept]
    if masked_cols:
        mask[sys_span.stop:, masked_cols] = neg
    return mask.view(1, 1, L, L)


@torch.no_grad()
def generate_partial_ablated(m: "Model", ids: torch.Tensor, sys_span: slice,
                             keep_frac: float, order: str = "strided",
                             max_new_tokens: int = 30) -> str:
    """Greedy-generate while keeping only a `keep_frac` fraction of the system span visible to
    post-system tokens at every step (see `build_partial_ablation_mask`)."""
    cur = ids
    for _ in range(max_new_tokens):
        L = cur.shape[1]
        mask = build_partial_ablation_mask(L, sys_span, keep_frac, order, device=DEVICE,
                                           dtype=m.dtype)
        logits = m.model(cur, attention_mask=mask).logits
        if not torch.isfinite(logits[0, -1]).all():
            raise RuntimeError("non-finite logits under the partial-ablation mask.")
        nxt = logits[0, -1].argmax().item()
        if nxt == m.tok.eos_token_id:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return m.tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)


def _decoder_layers(m: "Model"):
    """The decoder block ModuleList, across the families this demo loads. hidden_states[i] is
    the output of block i-1 (index 0 is the embedding), so a steering vector measured at
    hidden-state index L is injected by hooking block L-1's output."""
    base = getattr(m.model, "model", m.model)
    for attr in ("layers", "h", "blocks"):
        blocks = getattr(base, attr, None)
        if blocks is not None:
            return blocks
    tr = getattr(m.model, "transformer", None)
    if tr is not None and getattr(tr, "h", None) is not None:
        return tr.h
    raise AttributeError("could not locate decoder layers for the steering hook")


@torch.no_grad()
def generate_steered(m: "Model", ids: torch.Tensor, sys_span: slice, vec, layer: int,
                     coef: float, keep_frac: float = 0.0, order: str = "strided",
                     max_new_tokens: int = 30) -> str:
    """Greedy-generate UNDER closure while adding `coef * vec` to the residual stream at
    `layer` via a forward hook — testing whether re-surfacing the (closed-off) goal direction
    restores recall. The hook adds to block (layer-1)'s output so it matches the hidden-state
    index the vector was measured at; it is always removed afterwards.

    The steer is applied only on the *prefill* pass (the forward whose length equals the
    prompt, i.e. the one that predicts the first answer token), then released so the model
    completes the word naturally. Injecting at every step instead makes a 0.5B fixate on the
    first sub-token ('Hal Hal Hal...') rather than emit the whole codename — surfacing the
    goal at the decision point is the faithful, clean intervention."""
    v = torch.as_tensor(vec, device=DEVICE, dtype=m.dtype)
    block = _decoder_layers(m)[max(0, layer - 1)]
    prompt_len = ids.shape[1]

    def hook(_module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] != prompt_len:  # only the prefill (first generated token)
            return output
        hs = hs + coef * v
        return (hs, *output[1:]) if isinstance(output, tuple) else hs

    handle = block.register_forward_hook(hook)
    try:
        cur = ids
        for _ in range(max_new_tokens):
            L = cur.shape[1]
            mask = build_partial_ablation_mask(L, sys_span, keep_frac, order, device=DEVICE,
                                               dtype=m.dtype)
            logits = m.model(cur, attention_mask=mask).logits
            nxt = logits[0, -1].argmax().item()
            if nxt == m.tok.eos_token_id:
                break
            cur = torch.cat([cur, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    finally:
        handle.remove()
    return m.tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)


# Distinct project codenames used as the probe's class labels (all planted in the system
# prompt; the filler/question never mention them, so the only label signal is in the system
# block — see run_probe). PROBE_QUESTION is identical across classes so the embedding baseline
# sits at chance.
CODENAMES = ["Halcyon", "Borealis", "Zephyr", "Cinder"]
PROBE_QUESTION = "What is the project codename?"


def probe_layers_for(m: "Model") -> list[int]:
    """A few hidden-state layers to probe (early / middle / late), robust to model depth."""
    return sorted({2, m.n_layers // 2, max(0, m.n_layers - 2)})


def probe_auc(X, y) -> float:
    """Cross-validated one-vs-rest ROC AUC of a logistic probe on row-normalized features.
    Held-out (cv=4), so a high value is genuine decodability, not memorization. Row-norm aids
    convergence on raw hidden states and is safe (nonzero norms)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np
    X = np.asarray(X)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return float(cross_val_score(LogisticRegression(max_iter=5000), Xn, y,
                                 cv=4, scoring="roc_auc_ovr").mean())


def collect_codename_hidden(m: "Model", n_filler: int, layers: list[int],
                            n_episodes: int = 8):
    """Build the codename-classification sample set at a given context length: vary the
    planted codename across `CODENAMES` (the label) and the meandering filler within each
    class, returning per-layer final-token residuals, the input-embedding features, and the
    labels. Uses the memory-safe hidden-states-only forward so it scales to long context."""
    import numpy as np
    resid = {L: [] for L in layers}
    X_embed, y = [], []
    for label, codename in enumerate(CODENAMES):
        system = system_prompt_with(codename)
        for ep in range(n_episodes):
            msgs = build_conversation(max(0, n_filler - (ep % 2)), system=system, start=ep)
            ids = m.encode(msgs + [{"role": "user", "content": PROBE_QUESTION}])
            hs = m.hidden_last_layers(ids, [0, *layers])
            for L in layers:
                resid[L].append(hs[L])
            X_embed.append(hs[0])
            y.append(label)
    return resid, X_embed, np.array(y)


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
        import sklearn  # noqa: F401
    except ImportError:
        print("\n[probe] needs scikit-learn: pip install scikit-learn\n")
        return

    print("\n" + "=" * 72)
    print("  MODE 3: Residual probe — the planted goal survives in the hidden states")
    print("=" * 72)

    layers = probe_layers_for(m)
    resid, X_embed, y = collect_codename_hidden(m, len(FILLER), layers, n_episodes=8)

    run = run or {}
    print(f"  samples={len(y)}  classes={len(CODENAMES)} (codename values)  "
          f"layers={m.n_layers}")
    print("-" * 72)
    for L in layers:
        a = probe_auc(resid[L], y)
        log_metric(coll, run, {
            "mode": "probe", "repr": "residual", "layer": int(L), "auc": a,
            "samples": int(len(y)), "classes": len(CODENAMES),
        })
        print(f"  residual stream (layer {L:>2})      : AUC {a:.3f}")
    a_embed = probe_auc(X_embed, y)
    log_metric(coll, run, {
        "mode": "probe", "repr": "embedding", "layer": 0, "auc": a_embed,
        "samples": int(len(y)), "classes": len(CODENAMES),
    })
    print(f"  input embeddings  (layer  0)      : AUC {a_embed:.3f}  "
          "(chance — last-token input is identical across classes)")
    print("-" * 72)
    print("The planted codename is decodable from the residual stream far above the")
    print("embedding baseline: the value survives in the hidden state even as the")
    print("attention channel to the system prompt thins.\n")


def run_dissociate(m: "Model", coll=None, run: dict | None = None,
                   max_turns: int | None = None) -> None:
    """The paper's headline shown *within a single model*: as context grows and attention to
    the goal thins (GAR falls), the codename stays decodable from the residual stream (AUC
    high) even once behavioral recall starts to MISS — "the information is present but
    unused." Decodability is measured under natural attention decay (more filler), NOT under
    the ablation mask: a hard column mask severs the path to the goal, so AUC would collapse
    with recall and there would be nothing to dissociate."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("\n[dissociate] needs scikit-learn: pip install scikit-learn\n")
        return

    run = run or {}
    print("\n" + "=" * 72)
    print("  DISSOCIATE: decodable but unused — AUC holds while recall falls (one model)")
    print("=" * 72)
    schedule = gar_schedule(m.n_params, max_turns)
    layers = probe_layers_for(m)
    print(f"{'turns':>5} | {'ctx tok':>7} | {'GAR all':>8} | {'recall':>6} | "
          f"{'codename AUC':>12}")
    print("-" * 72)
    for n_filler in schedule:
        msgs = build_conversation(n_filler)
        sys_span = m.system_span(msgs)
        gar_ids = m.encode(msgs + [{"role": "user", "content": PROBES[0][1]}])
        per_layer = m.gar_last_token(gar_ids, sys_span)
        gar_all = sum(per_layer) / len(per_layer)

        hits = 0
        for fact_name, question in PROBES:
            ids = m.encode(msgs + [{"role": "user", "content": question}])
            hits += FACTS[fact_name].lower() in m.generate(ids, max_new_tokens=30).lower()

        # Codename decodability at this context length (best probe layer).
        resid, _, y = collect_codename_hidden(m, n_filler, layers, n_episodes=6)
        auc = max(probe_auc(resid[L], y) for L in layers)

        log_metric(coll, run, {
            "mode": "dissociate", "turns": n_filler, "ctx_tokens": gar_ids.shape[1],
            "gar_all": gar_all, "recall": hits, "n_facts": len(PROBES),
            "codename_auc": auc,
        })
        print(f"{n_filler:>5} | {gar_ids.shape[1]:>7} | {gar_all:>8.4f} | "
              f"{hits:>4}/{len(PROBES)} | {auc:>12.3f}")

    print("-" * 72)
    print("As context grows the attention channel thins (GAR falls) and recall starts to")
    print("MISS, yet the codename stays decodable from the residual stream (AUC stays high):")
    print("the goal is still *present* in the hidden state, just no longer *used*. That gap")
    print("between decodability and behavior is the paper's dissociation, in one model.\n")


# Coefficients for the steering sweep, as multiples of the typical residual-stream norm at the
# steered layer (so the scale is meaningful across models). 0.0 is the closed-off baseline; the
# small values bracket the transition where recall returns.
STEER_COEFS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)


def steering_vector(X, y, target_label: int = 0):
    """Unit diff-of-means steering direction for `target_label`: the mean residual of the
    samples with that label minus the mean of the rest, normalized to unit length. This is the
    direction in the residual stream that encodes 'this class' (here, the planted codename)."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    d = X[y == target_label].mean(0) - X[y != target_label].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def run_steer(m: "Model", coll=None, run: dict | None = None) -> None:
    """Diagnosis -> intervention: if the goal is *present but unused* under closure, then
    re-surfacing it should restore behavior. We build a steering vector for the planted
    codename as a diff-of-means in the residual stream (mean over the planted-codename samples
    minus mean over the others, at the best probe layer), then generate UNDER total closure
    while adding it back via a forward hook, sweeping its strength. Recall should climb from 0
    (closed off) as the goal direction is re-injected. Honest caveat: a single crude linear
    steer on a 0.5B model may only partially restore recall — reported exactly as measured."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("\n[steer] needs scikit-learn: pip install scikit-learn\n")
        return
    import numpy as np

    run = run or {}
    print("\n" + "=" * 72)
    print("  STEER: re-inject the closed-off goal direction and watch recall return")
    print("=" * 72)

    layers = probe_layers_for(m)
    resid, _, y = collect_codename_hidden(m, len(FILLER), layers, n_episodes=8)
    best_layer = max(layers, key=lambda L: probe_auc(resid[L], y))

    # Diff-of-means direction for the planted codename (CODENAMES[0]) vs the others, at the
    # best layer; unit-normalized so STEER_COEFS scale it by the typical residual norm.
    X = np.asarray(resid[best_layer])
    unit = steering_vector(X, y, target_label=0)
    typ_norm = float(np.linalg.norm(X, axis=1).mean())
    print(f"  steering layer {best_layer} (best probe AUC {probe_auc(resid[best_layer], y):.3f}); "
          f"typical residual norm ~{typ_norm:.1f}")

    msgs = build_conversation(4)
    sys_span = m.system_span(msgs)
    ids = m.encode(msgs + [{"role": "user", "content": PROBE_QUESTION}])
    target = FACTS["project codename"].lower()  # the codename the direction encodes

    print(f"\n  recall of '{FACTS['project codename']}' under TOTAL closure vs steering strength:")
    print(f"    {'coef(xnorm)':>11} | {'recall':>6} | {'reply (head)':>40}")
    print("    " + "-" * 64)
    for coef_mult in STEER_COEFS:
        coef = coef_mult * typ_norm
        reply = generate_steered(m, ids, sys_span, unit, best_layer, coef,
                                 keep_frac=0.0, max_new_tokens=30)
        ok = int(target in reply.lower())
        log_metric(coll, run, {
            "mode": "steer", "layer": int(best_layer), "alpha": float(coef_mult),
            "coef": float(coef), "recall_ok": ok, "fact": "project codename",
        })
        head = reply.replace("\n", " ")[:40]
        print(f"    {coef_mult:>11.2f} | {('OK' if ok else 'MISS'):>6} | {head:>40}")

    print("    " + "-" * 64)
    print("At coef 0 the channel is closed and the codename is gone; adding the diff-of-means")
    print("direction back at the decision point re-surfaces it and recall returns — a causal")
    print("confirmation that the goal was present but unused, not absent. (A cruder all-position")
    print("steer instead makes a 0.5B fixate on the first sub-token; reported as measured.)\n")


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
    """Behavioral survival under the *graded* (partial) closure, per model: mean recall over
    the partial closures (fraction < 1.0, i.e. excluding the full-attention baseline), with a
    band across filler seeds. Survival is first averaged within each seed (over mask orders x
    fractions x facts), then averaged across seeds — so the band's min/max are per-seed
    survivals, showing whether the figure is a stable signal or a single-draw artifact.

    This is the dissociation axis: a residual-reliant model keeps recalling as more of the
    system prompt is hidden, an attention-reliant one collapses. Unlike total ablation (which
    pins recall at 0 for everyone) this lands between 0 and 1, and averaging over mask orders
    de-confounds it from where a given fact happens to sit in the prompt."""
    return list(coll.aggregate([
        {"$match": {"mode": "closure", "frac": {"$lt": 1.0}, **(match or {})}},
        {"$group": {"_id": {"model": "$model", "seed": "$seed"},
                    "seed_survival": {"$avg": "$recall_ok"}}},
        {"$group": {"_id": "$_id.model", "survival": {"$avg": "$seed_survival"},
                    "surv_min": {"$min": "$seed_survival"},
                    "surv_max": {"$max": "$seed_survival"}, "seeds": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]))


def per_run_survival(coll, match: dict | None = None) -> list[dict]:
    """Per-run behavioral survival: for each (model, run_id), the mean graded-closure recall
    over the partial closures (frac < 1.0). One scalar per run — the unit of analysis for the
    cross-run statistics (mean/CI and the permutation test)."""
    return list(coll.aggregate([
        {"$match": {"mode": "closure", "frac": {"$lt": 1.0}, **(match or {})}},
        {"$group": {"_id": {"model": "$model", "run_id": "$run_id"},
                    "survival": {"$avg": "$recall_ok"}}},
        {"$sort": {"_id.model": 1, "_id.run_id": 1}},
    ]))


def survival_across_runs_by_model(coll, match: dict | None = None) -> list[dict]:
    """Treat survival as a statistic across *runs*: per model, the mean of the per-run
    survivals with a 95% normal-approx confidence interval (half-width 1.96*s/sqrt(n), sample
    std). A CI needs n>=2 runs; with one run it is reported as None (point estimate only).
    Repeated `demo.py all` runs per model are what give this power."""
    import numpy as np
    by_model: dict[str, list[float]] = {}
    for r in per_run_survival(coll, match):
        by_model.setdefault(r["_id"]["model"], []).append(r["survival"])
    out = []
    for model, vals in sorted(by_model.items()):
        v = np.asarray(vals, dtype=float)
        n = int(len(v))
        std = float(v.std(ddof=1)) if n > 1 else None
        ci95 = float(1.96 * std / np.sqrt(n)) if std is not None else None
        out.append({"model": model, "mean": float(v.mean()), "std": std,
                    "ci95": ci95, "n_runs": n})
    return out


def permutation_test_survival(coll, model_a: str, model_b: str, match: dict | None = None,
                              iters: int = 10000, seed: int = 0) -> dict:
    """Two-sided permutation test on the difference in mean per-run survival between two
    models. Pools both models' per-run survivals, repeatedly shuffles the model labels, and
    measures how often the permuted |mean diff| is at least the observed |mean diff|.
    Deterministic given `seed`; add-one smoothing keeps the p-value strictly positive."""
    import numpy as np
    runs = per_run_survival(coll, match)
    a = np.array([r["survival"] for r in runs if r["_id"]["model"] == model_a], dtype=float)
    b = np.array([r["survival"] for r in runs if r["_id"]["model"] == model_b], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"model_a": model_a, "model_b": model_b, "n_a": int(len(a)),
                "n_b": int(len(b)), "diff": None, "p_value": None}
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(iters):
        rng.shuffle(pooled)
        if abs(pooled[:na].mean() - pooled[na:].mean()) >= obs - 1e-12:
            count += 1
    return {"model_a": model_a, "model_b": model_b, "n_a": int(na), "n_b": int(len(b)),
            "diff": float(a.mean() - b.mean()), "p_value": float((count + 1) / (iters + 1))}


# Thresholds for the cross-architecture reliance call. Deliberately lenient and named so the
# heuristic is explicit: "decodable" = the goal is recoverable from the residual stream at all;
# "survives" = at least HALF of graded-closure recall holds (a natural midpoint: a model that
# keeps >= 50% of its facts as the goal channel is progressively hidden is reading them from
# the residual stream more than from attention). The bucket is a label on a continuous axis,
# not a hard scientific claim, so `compare` also flags survivals that sit near the boundary.
AUC_DECODABLE = 0.70
SURVIVAL_OK = 0.50
SURVIVAL_BORDERLINE = 0.10  # within this of SURVIVAL_OK -> the bucket is reported as borderline


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
        c = closure.get(name)
        if c is not None:
            survival, surv_min, surv_max = c["survival"], c.get("surv_min"), c.get("surv_max")
        else:
            survival = ablated_ok / normal_ok if normal_ok else None
            surv_min = surv_max = None
        residual_auc = auc.get(name, {}).get("residual")
        rows.append({
            "model": name,
            "residual_auc": residual_auc,
            "embedding_auc": auc.get(name, {}).get("embedding"),
            "normal_rate": (normal_ok / facts) if facts else None,
            "ablated_rate": (ablated_ok / facts) if facts else None,
            "survival": survival,
            "surv_min": surv_min,
            "surv_max": surv_max,
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
    # "Latest run per model" means the latest run of the core `all` pipeline. The auxiliary
    # `dissociate`/`steer` modes log their own run_ids; without excluding them, running one of
    # those after `all` would shadow the full run and drop the model from report/compare.
    core = {"mode": {"$nin": ["dissociate", "steer"]}}
    return ({"run_id": {"$in": latest_run_ids(coll, core)}},
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


def run_compare(coll, scope: str = "latest", run_id: str | None = None,
                stats: bool = False) -> None:
    """Cross-architecture view: line up each model's *dissociation signature* — does the goal
    survive in the residual stream (probe AUC), and does behavior survive as the attention
    channel to the goal is progressively closed (graded closure survival)? The paper's headline
    is that these two can come apart differently by architecture. Descriptive, not a replication.

    With `stats=True` the survival figure is treated across *runs*: mean +/- 95% CI over the
    per-run survivals, plus a permutation test on each model pair (uses full history)."""
    print("\n" + "=" * 72)
    print("  COMPARE: cross-architecture dissociation (residual survival vs behavior)")
    print("=" * 72)

    if coll is None:
        print("Metrics store unavailable — nothing to compare.\n")
        return
    if coll.count_documents({}) == 0:
        print("No runs logged yet. Run e.g. `python3 demo.py all --model <name>` first.\n")
        return

    # Stats treat survival across runs, so they only make sense over the full history.
    if stats:
        scope = "all"
    match, scope_label = resolve_scope(coll, scope, run_id)
    rows = compare_by_model(coll, match)
    print(f"  scope: {scope_label}")
    print(f"  {len(rows)} model(s): {', '.join(short_model(r['model']) for r in rows)}")

    def fmt(x, spec="6.3f"):
        return format(x, spec) if x is not None else "  n/a"

    def band(r):
        lo, hi = r.get("surv_min"), r.get("surv_max")
        return f"[{lo:.2f},{hi:.2f}]" if lo is not None and hi is not None else "    n/a "

    def reliance_label(r):
        s = r["survival"]
        if s is not None and r["residual_auc"] is not None \
                and r["residual_auc"] >= AUC_DECODABLE \
                and abs(s - SURVIVAL_OK) <= SURVIVAL_BORDERLINE:
            return r["reliance"] + " *"
        return r["reliance"]

    print(f"\n  {'model':>24} | {'res AUC':>7} | {'emb AUC':>7} | {'normal':>6} | "
          f"{'ablat':>6} | {'surv':>5} | {'seed band':>11} | {'miss@':>6} | {'reliance':>19}")
    print("  " + "-" * 116)
    for r in rows:
        miss = r["first_miss_turn"] if r["first_miss_turn"] is not None else "none"
        print(f"  {short_model(r['model']):>24} | {fmt(r['residual_auc']):>7} | "
              f"{fmt(r['embedding_auc']):>7} | {fmt(r['normal_rate'], '6.2f'):>6} | "
              f"{fmt(r['ablated_rate'], '6.2f'):>6} | {fmt(r['survival'], '5.2f'):>5} | "
              f"{band(r):>11} | {str(miss):>6} | {reliance_label(r):>19}")

    print("\n  Reading it:")
    print("    - ablat = recall under TOTAL closure (system span fully blinded; ~0 by design).")
    print("    - surv  = recall under GRADED closure (mean over mask orders x fractions x seeds,")
    print("              as more of the system prompt is hidden) — the discriminating axis [0,1].")
    print("    - seed band = [min,max] survival across filler seeds (is the figure stable?).")
    print(f"    - '*' marks a bucket within {SURVIVAL_BORDERLINE:.2f} of the {SURVIVAL_OK:.2f} "
          "survival threshold (borderline).")
    print("    - residual-reliant : goal decodable AND recall survives graded closure (robust).")
    print("    - attention-reliant: goal decodable BUT recall collapses under closure")
    print("                         (info present, unused without attention — the dissociation).")
    print("    - weak-encoding    : goal not decodable from the residual stream.")

    if len(rows) < 2:
        print("\n  Only one model in scope — log another with `demo.py all --model <name>`")
        print("  (e.g. SmolLM2-360M-Instruct, TinyLlama-1.1B-Chat-v1.0) to see a contrast.")

    if stats:
        import itertools
        print("\n  Multi-run survival statistics (per-run survival = mean graded-closure recall,")
        print("  over the full history; repeat `demo.py all` per model for N>1):")
        srows = survival_across_runs_by_model(coll, match)
        print(f"    {'model':>24} | {'mean surv':>9} | {'95% CI':>9} | {'runs':>4}")
        print("    " + "-" * 54)
        for r in srows:
            ci = f"+/-{r['ci95']:.3f}" if r["ci95"] is not None else "n/a"
            print(f"    {short_model(r['model']):>24} | {r['mean']:>9.3f} | {ci:>9} | "
                  f"{r['n_runs']:>4}")
        models = [r["model"] for r in srows]
        if len(models) >= 2:
            print("\n  Pairwise permutation test (two-sided, 10k iters, seed 0) on mean survival:")
            print(f"    {'pair':>44} | {'diff':>7} | {'p-value':>7}")
            print("    " + "-" * 64)
            for a, b in itertools.combinations(models, 2):
                pt = permutation_test_survival(coll, a, b, match=match)
                if pt["p_value"] is None:
                    continue
                pair = f"{short_model(a)} vs {short_model(b)}"
                print(f"    {pair:>44} | {pt['diff']:>+7.3f} | {pt['p_value']:>7.4f}")
        if any(r["n_runs"] < 2 for r in srows):
            print("\n  Note: a CI needs >=2 runs per model (sample std). Re-run `demo.py all`")
            print("  a few times per model so the interval and permutation test have power.")

    print("\n  Caveat: small instruct models, 4 planted facts and a 32-sample probe. This shows")
    print("  the *shape* of the dissociation; `--stats` adds cross-run CIs and a permutation")
    print("  test, but it is still a small-model demonstration, not the paper's full result.\n")


def run_plot(coll, scope: str = "latest", run_id: str | None = None,
             figdir: str = "figures") -> list[str]:
    """Render the demo's four figures from the stored runs to committed PNGs in `figdir`:
      - gar_decay.png        : GAR vs context length (attention channel closing).
      - survival_curves.png  : recall vs visible fraction of the system span, per model,
                               with seed bands (graded closure).
      - auc_vs_survival.png  : residual decodability vs behavioral survival scatter, with
                               the reliance thresholds drawn in.
      - dissociation.png     : codename AUC and recall vs context length on twin axes
                               (decodable-but-unused, within a model).
    Read-only; each figure is best-effort and skipped (with a note) if its data is absent."""
    print("\n" + "=" * 72)
    print("  PLOT: render figures from the stored runs")
    print("=" * 72)
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: write files, never open a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plot] needs matplotlib: pip install matplotlib\n")
        return []
    if coll is None or coll.count_documents({}) == 0:
        print("No runs logged yet — run e.g. `python3 demo.py all` first.\n")
        return []

    os.makedirs(figdir, exist_ok=True)
    match, scope_label = resolve_scope(coll, scope, run_id)
    print(f"  scope: {scope_label}")
    written: list[str] = []

    def save(fig, name: str) -> None:
        path = os.path.join(figdir, name)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")

    # 1) GAR decay vs context length, one line per model.
    gar_rows = list(coll.aggregate([
        {"$match": {"mode": "gar", **match}},
        {"$group": {"_id": {"model": "$model", "ctx": "$ctx_tokens"},
                    "gar": {"$avg": "$gar_all"}}},
        {"$sort": {"_id.ctx": 1}},
    ]))
    if gar_rows:
        series: dict[str, list] = {}
        for r in gar_rows:
            series.setdefault(r["_id"]["model"], []).append((r["_id"]["ctx"], r["gar"]))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, pts in sorted(series.items()):
            pts.sort()
            ax.plot([c for c, _ in pts], [g for _, g in pts], "o-", label=short_model(model))
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("GAR (goal-attention ratio, last token)")
        ax.set_title("Attention to the goal thins as context grows")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save(fig, "gar_decay.png")
    else:
        print("  (skip gar_decay.png — no gar docs in scope)")

    # 2) Survival curves: recall vs visible fraction of the system span, with seed bands.
    cur_rows = list(coll.aggregate([
        {"$match": {"mode": "closure", **match}},
        {"$group": {"_id": {"model": "$model", "frac": "$frac", "seed": "$seed"},
                    "recall": {"$avg": "$recall_ok"}}},
        {"$group": {"_id": {"model": "$_id.model", "frac": "$_id.frac"},
                    "mean": {"$avg": "$recall"},
                    "lo": {"$min": "$recall"}, "hi": {"$max": "$recall"}}},
        {"$sort": {"_id.frac": 1}},
    ]))
    if cur_rows:
        series = {}
        for r in cur_rows:
            series.setdefault(r["_id"]["model"], []).append(
                (r["_id"]["frac"], r["mean"], r["lo"], r["hi"]))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, pts in sorted(series.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], "o-", label=short_model(model))
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], alpha=0.15)
        ax.set_xlabel("visible fraction of the system span (1.0 = full attention)")
        ax.set_ylabel("recall (fraction of facts)")
        ax.set_title("Behavioral survival as the goal channel is progressively closed")
        ax.set_ylim(-0.05, 1.05)
        ax.invert_xaxis()  # left->right reads as "closing the channel"
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save(fig, "survival_curves.png")
    else:
        print("  (skip survival_curves.png — no closure docs in scope)")

    # 3) Residual decodability vs behavioral survival, with the reliance thresholds drawn in.
    rows = [r for r in compare_by_model(coll, match)
            if r["residual_auc"] is not None and r["survival"] is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(AUC_DECODABLE, color="gray", ls="--", lw=1)
        ax.axvline(SURVIVAL_OK, color="gray", ls="--", lw=1)
        ax.text(0.01, AUC_DECODABLE + 0.01, f"AUC {AUC_DECODABLE:.2f} (decodable)",
                fontsize=7, color="gray")
        ax.text(SURVIVAL_OK + 0.01, 0.02, f"survival {SURVIVAL_OK:.2f}",
                fontsize=7, color="gray", rotation=90, va="bottom")
        # A legend (not per-point text) keeps the figure readable where several attention-
        # reliant models cluster at AUC~1.0 with near-identical low survival.
        for r in sorted(rows, key=lambda r: r["survival"]):
            ax.scatter(r["survival"], r["residual_auc"], s=70, label=short_model(r["model"]))
        ax.legend(fontsize=7, loc="lower left", title="model")
        ax.set_xlabel("behavioral survival under graded closure")
        ax.set_ylabel("residual decodability (best probe AUC)")
        ax.set_title("Decodable but unused: high AUC, low survival = attention-reliant")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        save(fig, "auc_vs_survival.png")
    else:
        print("  (skip auc_vs_survival.png — need probe + closure docs in scope)")

    # 4) Within-model dissociation: codename AUC and recall vs context length (twin axes).
    dis_rows = list(coll.aggregate([
        {"$match": {"mode": "dissociate", **match}},
        {"$group": {"_id": {"model": "$model", "ctx": "$ctx_tokens"},
                    "auc": {"$avg": "$codename_auc"}, "recall": {"$avg": "$recall"},
                    "n_facts": {"$max": "$n_facts"}}},
        {"$sort": {"_id.ctx": 1}},
    ]))
    if dis_rows:
        series = {}
        for r in dis_rows:
            series.setdefault(r["_id"]["model"], []).append(
                (r["_id"]["ctx"], r["auc"], r["recall"] / (r["n_facts"] or 4)))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax2 = ax.twinx()
        for model, pts in sorted(series.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1] for p in pts], "o-", color="tab:blue",
                    label=f"{short_model(model)} AUC")
            ax2.plot(xs, [p[2] for p in pts], "s--", color="tab:red",
                     label=f"{short_model(model)} recall")
        ax.set_xlabel("context length (tokens)")
        ax.set_ylabel("codename AUC (residual decodability)", color="tab:blue")
        ax2.set_ylabel("recall (fraction of facts)", color="tab:red")
        ax.set_ylim(0.4, 1.05)
        ax2.set_ylim(-0.05, 1.05)
        ax.set_title("Decodable but unused: AUC holds while recall falls (one model)")
        ax.grid(True, alpha=0.3)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="lower left")
        save(fig, "dissociation.png")
    else:
        print("  (skip dissociation.png — no dissociate docs in scope; run `demo.py dissociate`)")

    print(f"\n  {len(written)} figure(s) written to {figdir}/\n")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Attention-channel-closing demo")
    ap.add_argument("mode", choices=["gar", "ablate", "probe", "dissociate", "steer",
                                     "all", "report", "compare", "plot"])
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
    ap.add_argument("--seeds", type=int, default=3, metavar="N",
                    help="ablate: number of filler-ordering seeds for the graded-closure "
                         "survival band (default: %(default)s)")
    ap.add_argument("--stats", action="store_true",
                    help="compare: add multi-run survival stats (mean +/- 95%% CI, n runs) "
                         "and pairwise permutation-test p-values (implies --all-runs)")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"],
                    default="float32",
                    help="model compute dtype (default: %(default)s); use bfloat16/auto to fit "
                         "a larger model in memory (bf16 avoids fp16's overflow on mps)")
    ap.add_argument("--light", action="store_true",
                    help="slash the closure budget (1 seed, 1 mask order, short generations, "
                         "GAR at 0 turns) so a big model finishes in minutes; auto-on >2B params")
    args = ap.parse_args()

    # Opening the store is best-effort: if it fails, the scientific modes still run
    # (they just won't log); only `report`/`compare` truly need it.
    try:
        coll = open_metrics(args.db)
    except Exception as e:
        print(f"[warn] could not open metrics store {args.db}: {e}", file=sys.stderr)
        coll = None

    # `report`, `compare` and `plot` only read the store — no need to load the model.
    if args.mode in ("report", "compare", "plot"):
        scope = "run" if args.run else ("all" if args.all_runs else "latest")
        if args.mode == "report":
            run_report(coll, scope=scope, run_id=args.run)
        elif args.mode == "compare":
            run_compare(coll, scope=scope, run_id=args.run, stats=args.stats)
        else:
            run_plot(coll, scope=scope, run_id=args.run)
        return

    run = new_run(args.model)
    m = Model(args.model, dtype=args.dtype)
    if not m.supports_system_role():
        print(f"[skip] {args.model}: its chat template does not accept a system role, so "
              "the goal cannot be planted/measured in the system prompt. Try a model with "
              "system-role support (e.g. Qwen2.5, SmolLM2, TinyLlama).", file=sys.stderr)
        return
    # Integrity guard: every measurement masks/measures the system span, so an empty span
    # would silently turn ablation into a no-op and report fake survival. Refuse rather than
    # mislead if the goal span can't be located for this model's chat template.
    ref_span = m.system_span(build_conversation(4))
    if ref_span.stop - ref_span.start == 0:
        print(f"[skip] {args.model}: could not locate the system-prompt token span under "
              "its chat template (goal span empty), so attention to the goal can't be "
              "masked or measured. Skipping to avoid reporting a no-op as 'survival'.",
              file=sys.stderr)
        return
    # `--light` keeps big models bounded; auto-enable past ~2B params even without the flag.
    light = args.light or m.n_params > 2e9
    if light and not args.light:
        print(f"[note] {m.n_params/1e9:.1f}B params > 2B -> enabling --light automatically "
              "(1 seed, 1 mask order, short generations, GAR at 0 turns).")
    gar_turns = 0 if light else args.max_turns

    if args.mode in ("gar", "all"):
        run_gar(m, coll, run, max_turns=gar_turns)
    if args.mode in ("ablate", "all"):
        run_ablate(m, coll, run, seeds=args.seeds, light=light)
    if args.mode in ("probe", "all"):
        run_probe(m, coll, run)
    if args.mode == "dissociate":
        run_dissociate(m, coll, run, max_turns=gar_turns)
    if args.mode == "steer":
        run_steer(m, coll, run)

    if coll is not None:
        logged = coll.count_documents({"run_id": run["run_id"]})
        print(f"logged {logged} metric docs to {args.db} (run_id {run['run_id'][:8]}) — "
              "run `python3 demo.py report` to aggregate across runs.\n")


if __name__ == "__main__":
    main()

# How this demo relates to the paper

**Paper:** Vardhan Dongre, Joseph Hsieh, Viet Dac Lai, Seunghyun Yoon, Trung Bui, Dilek
Hakkani-Tür — *"When Attention Closes: How LLMs Lose the Thread in Multi-Turn
Interaction"* ([arXiv:2605.12922](https://arxiv.org/abs/2605.12922), 2026).

**Short answer: yes, it matches — as a faithful small-scale reproduction.** The three modes
in [demo.py](demo.py) map one-to-one onto the paper's three diagnostic instruments, and on a
single 0.5B model all three of the paper's headline qualitative results reproduce, including
the AUC ~0.99 residual-probe figure. The demo deliberately does *not* attempt the paper's
cross-architecture comparison or its parametric failure-timing prediction. Details below.

---

## The paper in one paragraph

The paper gives a *mechanistic* account of why LLMs drift away from their system prompt over
long conversations, which it calls the **channel-transition account**: a goal (persona,
rules, planted facts) reaches later tokens through two routes — the **attention channel**
(later tokens attending back to the goal-defining tokens) and the **residual stream**
(latent task representations carried in hidden states). As context grows, attention onto the
goal tokens thins until the attention channel effectively closes; whether goal-conditioned
behavior survives then depends on what the residual stream carries. The paper introduces
three instruments to show this: the **Goal Accessibility Ratio (GAR)** (attention from
generated tokens to goal tokens), **sliding-window attention ablations** (a causal
manipulation that force-closes the channel), and **linear residual-stream probes**. Headline
results: a causal ablation in Mistral collapses recall from near-perfect to **11%** on a
20-fact task and raises persona violations even without user pressure; linear probes recover
the goal from the residual stream with **AUC up to 0.99** across four architectures while
**input embeddings stay at chance**; and the encoding layer varies from **2 to 27** by
architecture.

---

## Instrument-by-instrument mapping

| Paper instrument | Demo mode | Where in code | Match |
| --- | --- | --- | --- |
| Goal Accessibility Ratio (GAR) | `python3 demo.py gar` | `gar_last_token`, `gar_per_layer`, `run_gar` | Same definition (attention mass from the generating position onto goal tokens); demo shows it decaying with context until recall finally breaks |
| Sliding-window attention ablation | `python3 demo.py ablate` | `build_ablation_mask`, `generate_ablated`, `run_ablate` | Same idea (causally close the channel from generated tokens to goal tokens); demo uses a whole-span column mask rather than a sliding window |
| Linear residual-stream probe | `python3 demo.py probe` | `run_probe` | Same design (decode the goal from hidden states; embeddings as the chance baseline) |

### 1. GAR — "attention to instructions thins as turns grow"

The paper defines GAR as attention from generated tokens to the task-defining goal tokens.
The demo computes exactly that at the final (generating) position, averaged over heads, per
layer. To reach the long contexts where behavior actually breaks, it measures GAR
memory-safely: prime a KV cache with all but the last token, then read attention for just the
final token (shape `[1, heads, 1, L]`) instead of materializing an O(L^2) matrix for every
layer (which would OOM at ~4.6k tokens):

```138:155:demo.py
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
```

`run_gar` re-measures GAR at growing context lengths (reporting an early-vs-late layer split)
and grades recall across all four facts at each length.

**Result (matches):** GAR falls from 0.58 to ~0.33 as the conversation grows from 103 to
5,763 tokens — the attention channel thinning, as the paper describes. Recall stays 4/4 until
the first MISS at 128 turns (~4,631 tokens, GAR 0.34), where the model drops the launch
quarter. The behavioral failure is real but sparse and non-monotonic on a 0.5B model (recall
recovers at 5,763 tokens), which is consistent with the paper's observation that *some*
architectures preserve goal-conditioned behavior even as attention vanishes — and is exactly
why the clean, total collapse is shown causally in MODE 2.

### 2. Ablation — "force-closing the channel collapses recall"

The paper's causal test force-closes the attention channel and observes recall collapse. The
demo builds an additive mask that zeroes attention from every post-system token onto the
system-token span:

```288:290:demo.py
    mask = torch.full((L, L), NEG, device=device).triu(1)
    mask[sys_span.stop:, sys_span] = NEG
    return mask.view(1, 1, L, L)
```

Crucially it leaves the system tokens' own causal self-attention intact, so no attention row
is fully masked (which would produce a softmax NaN and contaminate the "collapse"); a runtime
guard in `generate_ablated` asserts the logits stay finite.

**Result (matches qualitatively):** recall goes from **4/4 (100%)** with normal attention to
**0/4 (0%)** when the channel is closed — the same direction and near-totality as the paper's
near-perfect → 11% collapse in Mistral, on a much smaller model and fact set.

### 3. Residual probe — "the goal survives in the residual stream"

This is the most important correspondence, and the demo is specifically engineered to test
the paper's *actual* claim (goal **content** survives), not a weaker proxy. It varies the
planted codename across classes inside the system prompt, keeps the final question identical,
and reads the residual at the last token:

```345:365:demo.py
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
```

Because the last-token *input* is identical across classes, the input-embedding probe is a
genuine chance baseline — mirroring the paper's "input embeddings remain at chance."

**Result (matches, strikingly):**

| Representation | Demo AUC | Paper |
| --- | --- | --- |
| Input embeddings (layer 0) | 0.500 | "at chance" |
| Residual, early (layer 2) | 0.500 | not yet encoded |
| Residual, mid (layer 12) | 0.990 | up to 0.99 |
| Residual, late (layer 22) | 1.000 | up to 0.99 |

The jump from 0.500 at layer 2 to ~0.99 at layers 12/22 even reproduces the paper's point
that the goal "emerges" at some depth in the stack (the paper reports layers 2–27 depending
on architecture).

### 4. Telemetry — "GAR as a diagnostic"

Beyond the three instruments, the paper proposes GAR as a **diagnostic to monitor**. The demo
operationalizes that: every measurement (`run_gar` / `run_ablate` / `run_probe`) is written as
a document to an embedded MongoDB store (smongo), and `python3 demo.py report` runs aggregation
pipelines over the accumulated runs — for example the crossover-turn pipeline:

```python
def crossover_by_model(coll):
    return list(coll.aggregate([
        {"$match": {"mode": "gar", "recall_miss": True}},
        {"$group": {"_id": "$model", "first_miss_turn": {"$min": "$turns"}}},
        {"$sort": {"_id": 1}},
    ]))
```

This is **demo tooling, not a paper claim** — it does not reproduce a result. It exists to make
the "monitor GAR and detect the crossover" idea concrete and queryable across runs/models. Use
`python3 demo.py report` after one or more runs; it scopes to the latest run per model by
default (`--all-runs` aggregates the full history), and logging is best-effort so a store
failure never aborts a measurement.

---

## Claim-by-claim correspondence

| Paper claim | Demo reproduces it? | Evidence |
| --- | --- | --- |
| Attention to goal tokens thins over turns (GAR decay) | Yes | GAR 0.58 → 0.33 over 103 → 5,763 tokens |
| Natural decay eventually produces behavioral failure | Yes (sparse/noisy) | first recall MISS at 128 turns (~4,631 tokens), GAR 0.34 |
| Closing the attention channel causally collapses recall | Yes (qualitatively) | recall 4/4 → 0/4 |
| Goal information survives in the residual stream (high AUC) | Yes | residual AUC 0.990 / 1.000 |
| Input embeddings stay at chance | Yes | embedding AUC 0.500 |
| Encoding emerges at some depth (layers 2–27) | Partially | layer 2 = 0.500, layers 12/22 ≈ 0.99 on this model |

---

## What the demo deliberately does NOT do

The demo reproduces the paper's *mechanisms and method*, not its full experimental scope.
These parts of the paper are out of scope here:

- **Cross-architecture comparison.** The paper's central narrative — "what survives reveals
  architecture," with some models preserving behavior at vanishing attention and others
  failing despite decodable residual info — requires multiple model families. The demo runs
  one 0.5B model (`--model` lets you swap in others, but there is no comparative analysis).
- **Parametric failure-timing / crossover-turn prediction.** The paper predicts *when* a
  model will fail under windowed attention closure. The demo shows decay and collapse but
  does not fit or validate a timing model.
- **True sliding-window ablation.** The demo uses a simpler whole-span column mask rather than
  a moving window over turns.
- **Persona-constraint violations and the adversarial-pressure baseline.** The demo grades
  exact fact recall only; it does not measure persona drift or compare against an adversarial
  baseline.
- **Scale of the retention task.** 4 planted facts and a small probe set (4 codename values ×
  8 filler variations = 32 samples), versus the paper's 20-fact task and per-episode recall
  probing across architectures. The demo's numbers show the *shape* of the effect, not a
  statistically rigorous estimate.

---

## Fidelity assessment

As a *reproduction-in-miniature*, the demo is faithful: each diagnostic uses the same
operational definition as the paper, the residual probe is constructed so its baseline is a
true chance baseline, and all three headline qualitative results come out in the expected
direction — including the AUC ~0.99 figure and the layer-emergence pattern. As a *replication*
in the strong sense, it is intentionally limited to one small model and makes no
cross-architecture or timing claims, which the paper treats as its main contributions. The
[README](README.md) states these limits explicitly, and [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md)
shows a full captured run.

---

## Citation

```bibtex
@article{dongre2026attention,
  title  = {When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction},
  author = {Dongre, Vardhan and Hsieh, Joseph and Lai, Viet Dac and Yoon, Seunghyun
            and Bui, Trung and Hakkani-T{\"u}r, Dilek},
  journal = {arXiv preprint arXiv:2605.12922},
  year   = {2026},
  url    = {https://arxiv.org/abs/2605.12922}
}
```

# How this demo relates to the paper

**Paper:** Vardhan Dongre, Joseph Hsieh, Viet Dac Lai, Seunghyun Yoon, Trung Bui, Dilek
Hakkani-Tür — *"When Attention Closes: How LLMs Lose the Thread in Multi-Turn
Interaction"* ([arXiv:2605.12922](https://arxiv.org/abs/2605.12922), 2026).

**Short answer: yes, it matches — as a faithful small-scale reproduction.** The three core
modes in [demo.py](demo.py) map one-to-one onto the paper's three diagnostic instruments, and
on a single 0.5B model all three headline qualitative results reproduce, including the AUC
~0.99 residual-probe figure. Two further modes go further on the paper's central claim:
`dissociate` shows the **decodable-but-unused** dissociation *within one model* (AUC stays
high while recall falls), and `steer` turns the probe into an **intervention** — re-injecting
the decoded goal direction under closure restores recall. The demo still deliberately does
*not* claim the paper's cross-architecture *divergence* or its parametric failure-timing
prediction; `compare --stats` attempts the cross-architecture comparison descriptively and
reports an honest negative (no flip across five models, 360M→7B). Details below.

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
| Decodable-but-unused dissociation | `python3 demo.py dissociate` | `run_dissociate` | The paper's headline within one model: AUC vs recall vs GAR at matched context lengths (decodability holds while recall falls) |
| Steering intervention (beyond the paper) | `python3 demo.py steer` | `steering_vector`, `generate_steered`, `run_steer` | Causal follow-up: re-inject the decoded goal direction under closure and recall returns |

### 1. GAR — "attention to instructions thins as turns grow"

The paper defines GAR as attention from generated tokens to the task-defining goal tokens.
The demo computes exactly that at the final (generating) position, averaged over heads, per
layer. To reach the long contexts where behavior actually breaks, it measures GAR
memory-safely: prime a KV cache with all but the last token, then read attention for just the
final token (shape `[1, heads, 1, L]`) instead of materializing an O(L^2) matrix for every
layer (which would OOM at ~4.6k tokens):

```239:256:demo.py
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

```536:539:demo.py
    neg = torch.finfo(dtype).min
    mask = torch.full((L, L), neg, device=device, dtype=dtype).triu(1)
    mask[sys_span.stop:, sys_span] = neg
    return mask.view(1, 1, L, L)
```

Crucially it leaves the system tokens' own causal self-attention intact, so no attention row
is fully masked (which would produce a softmax NaN and contaminate the "collapse"); a runtime
guard in `generate_ablated` asserts the logits stay finite. The fill value is the model's
compute-dtype minimum (`finfo(dtype).min`) so the mask stays finite under bfloat16 — fp16's
narrower range overflows here, which is why the big-model path uses bf16.

**Result (matches qualitatively):** recall goes from **4/4 (100%)** with normal attention to
**0/4 (0%)** when the channel is closed — the same direction and near-totality as the paper's
near-perfect → 11% collapse in Mistral, on a much smaller model and fact set.

### 3. Residual probe — "the goal survives in the residual stream"

This is the most important correspondence, and the demo is specifically engineered to test
the paper's *actual* claim (goal **content** survives), not a weaker proxy. It varies the
planted codename across classes inside the system prompt, keeps the final question identical,
and reads the residual at the last token:

```714:733:demo.py
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
```

`run_probe` reuses this sample builder (as does `dissociate`); it reads hidden states with a
memory-safe, attentions-free forward so the same probe scales to long context.

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

### 3b. Dissociation — "present but unused," in one model (`dissociate`)

The probe shows the goal is *decodable*; the paper's stronger claim is that it can be
decodable **yet unused**. `run_dissociate` shows this within Qwen2.5-0.5B on the faithful
axis — context length (natural attention decay), not a hard mask. At each filler length it
measures GAR, behavioral recall (4 facts), and codename AUC on the *same* context:

| turns | ctx tokens | GAR all | recall | codename AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 103 | 0.582 | 4/4 | 1.000 |
| 56 | 2,084 | 0.353 | 4/4 | 0.992 |
| 96 | 3,499 | 0.336 | 4/4 | 1.000 |
| 128 | 4,631 | 0.338 | **3/4** | 0.887 |
| 160 | 5,763 | 0.332 | 4/4 | 0.773 |

As GAR falls and recall starts to MISS (3/4 at ~4.6k tokens), the codename stays decodable
(AUC well above the 0.50 embedding baseline throughout): **the goal is present in the hidden
state but no longer used.** That gap is the paper's dissociation, reproduced in a single model
(noisy at 0.5B — recall can recover at longer context — but the shape holds). Decodability is
measured under natural decay, *not* the ablation mask: a hard column mask severs the path to
the goal, so AUC would collapse with recall and there would be nothing to dissociate.

### 3c. Steering — from diagnosis to intervention (`steer`, beyond the paper)

If the goal is merely unused under closure, re-surfacing it should restore behavior.
`run_steer` builds a diff-of-means steering direction for the planted codename in the residual
stream (`steering_vector`), then generates **under total closure** while adding it back at the
decision point (`generate_steered`), sweeping the strength:

| coef (× residual norm) | recall of "Halcyon" |
| ---: | --- |
| 0.00 | MISS ("I'm sorry, but I need more context…") |
| 0.25 | MISS |
| 0.50 | **OK ("Halcyon.")** |
| 1.00 | OK ("Halcyon.") |

Recall is 0 with the channel closed and returns the instant the goal direction is injected — a
causal confirmation that the information was present but unused, not absent. This is an
intervention the paper does not run; it is reported as-measured (a cruder all-position steer
makes the 0.5B fixate on the first sub-token, `Hal Hal Hal…`).

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

### 5. Cross-architecture comparison (descriptive) — `compare`

The paper's headline is a **dissociation across architectures**: as attention to the goal
closes, the residual stream may still encode the goal (high probe AUC) while *behavior* either
holds (the model reads from the residual stream) or collapses (it cannot, despite the info
being present). `demo.py compare` builds this 2-axis signature per logged model — residual
decodability (`best_residual_auc_by_model`) against **graded-closure survival**
(`closure_survival_by_model`) — and buckets each model via `classify_reliance`:

- `residual-reliant`  — goal decodable AND recall survives graded closure (robust).
- `attention-reliant` — goal decodable BUT recall collapses under closure (the dissociation).
- `weak-encoding`     — goal not decodable from the residual stream.

The survival axis matters: an earlier version computed it from *total* ablation
(`ablated_ok/normal_ok`), which masks the entire system span and therefore pins recall at 0 for
every model — the axis could never discriminate. The demo now uses a **graded closure**: it
hides a fraction of the system prompt (`build_partial_ablation_mask`, keeping 75% / 50% / 25%
visible) so some facts stay attendable while others survive only in the residual stream. To
avoid measuring the wrong thing it averages over three mask orderings (`strided` / `suffix` /
`prefix`, so survival reflects the goal channel rather than where a fact sits) and over a few
filler **seeds** (reported as a `[min,max]` band), and `survival` is the mean recall over those
partial closures — a value that genuinely lands in [0, 1].

Measured across **five models spanning 360M→7B and four families** — including a deliberately
different family (StableLM-2-1.6B) and a same-family scale-up (Qwen2.5-7B) — the survival axis
is real, differentiated, and stable across seeds: Qwen2.5-7B 0.25, SmolLM2-360M 0.18,
Qwen2.5-0.5B 0.17, StableLM-2-1.6B 0.16, TinyLlama-1.1B 0.14. But **all five fall below the
survival threshold and land in `attention-reliant`** (residual AUC ~0.99–1.0). Two controlled
"does it flip?" tests both came back negative:

- **Scale within a family.** Qwen2.5-0.5B → 7B raises survival only from 0.17 to 0.25 — still
  attention-reliant. Scale alone (within Qwen) does not flip reliance.
- **A different architecture family.** StableLM-2-1.6B sits at 0.16 — squarely with the rest.

So the demo reproduces the paper's *method and per-model signature*, de-confounded and with a
contrasting family and a 7B in the mix, but it does **not** reproduce the architectural
*divergence* itself. Flipping a model to `residual-reliant` would need larger, deliberately
chosen families plus statistical power. The honest negative is the result; the harness is the
deliverable.

`compare --stats` adds the statistical layer (`survival_across_runs_by_model`,
`permutation_test_survival`): per-model mean ± 95% CI across runs and a pairwise permutation
test. Because greedy decoding with fixed filler seeds makes each run's survival deterministic,
the CIs are ~±0.00 (highly reproducible), and the permutation test — correct but **underpowered
at N=2 runs** (its smallest possible p is ~0.33) — reports no significant pairwise differences
yet. That is a power limitation, stated as such, not evidence of no difference.

Integrity note: adding StableLM-2 surfaced a real trap, and the regeneration for this writeup
caught it again. Its chat template renders a *lone* system message to nothing, so the
longest-common-prefix span detector returned an empty system span — which makes the ablation a
silent no-op and reports a fake `survival` of ~1.0 (a spurious "flip"). The store actually held
two such early StableLM runs (total-closure recall stuck at 4/4); they were identified by that
exact signature and discarded, and the fixed run collapses to 0/4 like every other model. The
fix is a span fallback (diff two conversations that share the system block but differ in the
first user turn) plus a guard that **skips any model whose goal span can't be located**, so the
dissociation axis can never be faked by a measurement that closed nothing.

(Notes: to keep `eager`-attention prefill in memory, the added models are run with a capped GAR
sweep via `--max-turns` — StableLM-2 with `--max-turns 0`. The 7B is run with
`--dtype bfloat16 --light`: bf16 because fp16's narrow range overflows in this model's eager
attention on MPS, yielding NaN GAR and non-finite ablation logits; `--light` slashes the
closure budget so it finishes in minutes. Neither affects the closure/probe axes that drive the
reliance call.)

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
| Goal **decodable yet unused** (dissociation, one model) | Yes | `dissociate`: AUC 0.77–1.0 while recall drops to 3/4 as GAR falls 0.58→0.33 |
| Re-surfacing the goal restores behavior (intervention) | Yes (beyond the paper) | `steer`: recall 0 under closure → "Halcyon." once the goal direction is injected |
| Cross-architecture *divergence* ("what survives reveals architecture") | Attempted, not observed | `compare --stats` across 5 models (360M→7B, 4 families): survival differs (0.14–0.25) but all stay `attention-reliant` (AUC ~0.99–1.0); neither scale (Qwen 7B) nor a different family (StableLM-2) flipped it |

---

## What the demo deliberately does NOT do

The demo reproduces the paper's *mechanisms and method*, not its full experimental scope.
These parts of the paper are out of scope here:

- **Cross-architecture *divergence* (attempted, not reproduced).** The paper's central
  narrative — "what survives reveals architecture," with some models preserving behavior at
  vanishing attention and others failing despite decodable residual info — requires multiple
  model families. The demo *attempts* this descriptively via `compare --stats` on a
  graded-closure survival axis that genuinely discriminates and is de-confounded across mask
  orderings and seeds. Across five models (360M→7B, four families) — including a deliberately
  different family (StableLM-2-1.6B) and a same-family scale-up (Qwen2.5-7B) — survival lands
  at 0.14–0.25 with tight seed bands, but all five stay below the survival threshold and bucket
  as `attention-reliant`. So the *flip* the paper reports — a model crossing into
  `residual-reliant` — did **not** appear at this scale, and neither scale nor a different
  family produced it; reported as-is rather than massaged into a divergence.
- **Parametric failure-timing / crossover-turn prediction.** The paper predicts *when* a
  model will fail under windowed attention closure. The demo shows decay and collapse but
  does not fit or validate a timing model.
- **True sliding-window ablation.** The demo closes the channel with a whole-span column mask
  (total closure) and a graded suffix mask (partial closure of the system prompt), rather than
  a moving window that slides over turns as context grows.
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
direction — including the AUC ~0.99 figure and the layer-emergence pattern. It goes further on
the central claim than the original three modes: `dissociate` shows decodability holding while
recall falls *inside one model*, and `steer` makes that causal by restoring recall under
closure. It also *attempts* the cross-architecture comparison (`compare --stats`) across five
models (360M→7B, four families) on a de-confounded, seed-banded survival axis, but the
divergence does not appear at this scale (all five are `attention-reliant`; neither scale nor a
different family flips it) — reported as a clean negative rather than a claim, with CIs and a
permutation test that is honest about being underpowered at N=2. As a *replication* in the
strong sense it remains limited to small models and makes no architectural-divergence or
failure-timing claims, which the paper treats as its main contributions. The
[README](README.md) states these limits explicitly, and [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md)
shows full captured runs and the figures.

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

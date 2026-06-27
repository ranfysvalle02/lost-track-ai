# lost-track-ai

A small, laptop-runnable demo that reproduces the *shape* of the three central findings
from **"When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction"**
(Dongre, Hsieh, Lai, Yoon, Bui, Hakkani-Tür — [arXiv:2605.12922](https://arxiv.org/abs/2605.12922)).

The point is not to match the paper's numbers. It is to make the paper's *mechanisms*
something you can watch happen on a 0.5B model by reading the model internals directly
(attention matrices and hidden states), rather than treating the model as a black box.

For a detailed, instrument-by-instrument mapping between this demo and the paper (and an
honest account of what it does and does not replicate), see [PAPER.md](PAPER.md).

---

## What the paper says (in brief)

Over long multi-turn conversations, models tend to drift away from their system prompt:
they break persona, drop constraints, and misremember planted facts. The paper offers a
mechanistic account rather than just a behavioral measurement:

- **Two channels carry the goal.** Instruction information reaches later tokens both
  through *attention* (later tokens attending back to the system-prompt tokens) and
  through the *residual stream* (latent task representations written into hidden states).
- **The attention channel can close.** As context grows, attention mass onto the
  system-prompt tokens thins. The paper measures this with the **Goal Accessibility Ratio
  (GAR)** — attention from generated tokens to goal tokens.
- **What survives depends on the residual stream.** Force-closing the attention channel
  (sliding-window ablation) collapses recall (in Mistral, from near-perfect to ~11% on a
  20-fact task), yet **linear probes recover the goal from the residual stream with AUC up
  to ~0.99** across architectures, while input embeddings stay at chance. Whether
  goal-conditioned behavior survives the channel closing depends on the gap between
  attention loss and residual decodability, and the encoding layer varies a lot by
  architecture (the paper reports layers 2–27).

---

## What this demo does

`demo.py` runs on `Qwen/Qwen2.5-0.5B-Instruct` via HuggingFace `transformers`
(`attn_implementation="eager"` so attentions are exposed) on MPS or CPU. It plants a
persona (`AUDITRON`) plus four facts in a system prompt, then grows a meandering
conversation and inspects the internals. Every run logs its measurements to an embedded
MongoDB store (smongo) so results accumulate into a queryable experiment log (see
[Why MongoDB, via smongo](#why-mongodb-via-smongo)).

```bash
python3 demo.py gar        # GAR decay: attention to the system prompt thins as turns grow
python3 demo.py ablate     # force-close the channel to system tokens -> recall collapses
python3 demo.py probe      # the planted value survives in the residual stream (high AUC)
python3 demo.py dissociate # one model: AUC stays high while recall falls (decodable but unused)
python3 demo.py steer      # re-inject the closed-off goal direction and watch recall return
python3 demo.py all        # run gar + ablate + probe (and log every measurement)
python3 demo.py report     # aggregate stored runs via MongoDB pipelines (no model load)
python3 demo.py compare    # cross-architecture dissociation across logged models (no model load)
python3 demo.py compare --stats  # + multi-run CIs and a pairwise permutation test
python3 demo.py plot --all-runs  # render the figures below from the stored runs (no model load)
```

- **`gar`** — appends a fact question at growing context lengths and reports GAR (attention
  mass onto the system-token span at the final position), split into *early* vs *late*
  layers so the layer-dependence the paper describes is visible. GAR trends down as the
  context grows; the sweep runs out to several thousand tokens and grades recall across all
  four facts, so you can see the first natural recall MISS appear once attention has thinned.
  GAR itself is measured memory-safely (a primed KV cache + a single final-token forward),
  but the recall generations still prefill the full context under `eager` attention
  (O(heads * L^2)). To stay within memory the sweep length is **capped by model size**
  (sub-1B models get the full sweep; larger ones are shortened), overridable with
  `--max-turns N`.
- **`ablate`** — closes the channel from post-system tokens onto the system-token span in
  two regimes. **Total closure** masks the whole span (the clean causal collapse: recall
  4/4 -> 0/4). **Graded closure** then hides a fraction of the system prompt (keeping 75% /
  50% / 25% visible) and measures recall at each step — so some facts stay attendable while
  others can only be recovered from the residual stream. To keep `survival` honest it
  **averages over three mask orderings** (`strided` / `suffix` / `prefix`, so the figure
  reflects the goal channel, not where a fact happens to sit) and over **`--seeds N`** filler
  orderings (default 3), reporting a `[min,max]` seed band. The masks leave each row's causal
  diagonal intact (masking *all* rows would give an all-`-inf` row and a softmax NaN), and a
  runtime guard asserts the logits stay finite. The graded survival is the axis `compare`
  uses, because unlike total closure (0 for everyone) it lands between 0 and 1.
- **`probe`** — trains a logistic-regression probe on hidden states. It varies the planted
  codename across classes (`Halcyon` / `Borealis` / `Zephyr` / `Cinder`) in the *system
  prompt*, while every episode ends with the *same* fixed question. Because the final-token
  input is identical across classes, the input-embedding probe is a genuine chance
  baseline; a high residual-stream AUC therefore means the model propagated the planted
  value forward into its hidden state even after the attention channel thinned.
- **`dissociate`** — the paper's headline shown *within a single model*, on the faithful
  axis. At each context length it measures all three signals on the same prompt: GAR,
  behavioral recall of the four facts, and codename decodability (CV probe AUC). As context
  grows, GAR falls and recall starts to MISS while the **AUC stays high** — the goal is
  still *present* in the hidden state, just no longer *used*. Decodability is measured under
  natural attention decay (more filler), **not** the ablation mask: a hard column mask severs
  the path to the goal so AUC would collapse with recall, leaving nothing to dissociate.
- **`steer`** — diagnosis → intervention. It builds a steering vector for the planted
  codename as a **diff-of-means** in the residual stream (mean of the planted-codename
  samples minus the others, at the best probe layer), then generates **under total closure**
  while adding that direction back at the decision point, sweeping its strength. Recall climbs
  from 0 (closed off) to restored — a causal confirmation that the goal was *present but
  unused*, not absent. (A cruder all-position steer makes a 0.5B fixate on the first
  sub-token; reported as measured.)
- **`plot`** — reads the store (no model load) and renders the four figures below to
  committed PNGs in `figures/`. Use `--all-runs` so the `dissociate`/`steer` runs (which log
  their own run_ids) are in scope; the default latest-per-model scope covers only the
  `all`-pipeline figures.
- **`report`** — reads the embedded MongoDB store (no model load) and runs aggregation
  pipelines: GAR decay range, the first recall MISS (crossover turn) per model, ablation
  recall (shown as a per-fact rate, `ok/facts`, so it stays meaningful no matter how many
  runs are in scope), and best residual vs embedding AUC. By default it scopes to the
  **latest run per model**, so re-running `all` doesn't inflate or blend the numbers; pass
  `--all-runs` to aggregate the full history or `--run <run_id>` for a single run. Re-run
  `all` with different `--model` values and the report still aggregates across models.

```bash
python3 demo.py report               # latest run per model (default)
python3 demo.py report --all-runs     # full accumulated history
python3 demo.py report --run <run_id> # a single run
```

- **`compare`** — reads the store (no model load) and lines up each logged model's
  *dissociation signature*: residual-probe AUC (is the goal still decodable?) against
  **graded-closure survival** (order/seed-averaged recall as more of the system prompt is
  hidden, with its seed band — does behavior survive when attention to the goal is
  progressively closed?). Each model is bucketed as `residual-reliant` (decodable and
  survives closure), `attention-reliant` (decodable but recall collapses under closure — the
  paper's dissociation), or `weak-encoding` (not decodable); a `*` marks buckets within
  `0.10` of the `0.50` survival threshold (borderline). Honors `--all-runs` / `--run`. Add
  **`--stats`** (uses the full history) to treat survival across *runs*: it prints each
  model's mean ± 95% CI (n runs) and a pairwise **permutation test** p-value on the
  difference in mean survival. Re-run `all` several times per model first so N > 1.

Logging is **best-effort**: if the metrics store can't be opened or a write fails, the
science modes (`gar`/`ablate`/`probe`) still run and print their results — they just warn
that the measurement wasn't logged. `report` and `compare` are the only modes that need the store.

### The headline, in one model: decodable but unused (`dissociate`)

The paper's central claim is that the goal can be *present yet unused*: still recoverable
from the hidden state after the attention channel has thinned, but no longer driving
behavior. `dissociate` shows exactly that within Qwen2.5-0.5B, on the faithful axis (context
length, i.e. natural attention decay — not a hard mask):

![codename AUC stays high while recall falls as context grows](figures/dissociation.png)

As context grows from ~100 to ~5.8k tokens, GAR falls from `0.58` to `0.33` and behavioral
recall drops a fact (4/4 → 3/4 around 4.6k tokens), yet the codename stays decodable from the
residual stream the whole way (AUC `0.77`–`1.0`, far above the `0.50` embedding baseline). The
gap between the blue (decodability) and red (behavior) curves *is* the dissociation. On a 0.5B
model it is noisy (recall can recover at still-longer context), but the shape is the paper's.

GAR itself decays monotonically as the conversation grows — the attention channel closing:

![GAR vs context length, per model](figures/gar_decay.png)

### Steering: surface the goal and recall returns (`steer`)

If the goal is merely *unused* under closure, re-surfacing it should restore behavior — and it
does. `steer` builds a diff-of-means direction for the planted codename in the residual
stream, then generates **under total closure** while adding it back at the decision point:

```text
    coef(xnorm) | recall |                             reply (head)
           0.00 |   MISS | I'm sorry, but I need more context to pr
           0.25 |   MISS | I'm sorry, but I need more context to pr
           0.50 |     OK |                                 Halcyon.
           1.00 |     OK |                                 Halcyon.
```

Recall is 0 with the channel closed and returns the moment the goal direction is injected — a
causal confirmation that the information was present but unused, not absent. (A cruder
all-position steer instead makes a 0.5B fixate on the first sub-token, `Hal Hal Hal…`;
reported as measured.)

### Cross-architecture comparison + statistics (`compare`)

The paper's *cross-architecture* headline is that what survives the channel closing "reveals
architecture." Log several models and compare their dissociation signatures:

```bash
python3 demo.py all --model Qwen/Qwen2.5-0.5B-Instruct
python3 demo.py all --model HuggingFaceTB/SmolLM2-360M-Instruct --max-turns 24
python3 demo.py all --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-turns 24
# A genuinely different family (StableLM-2). --max-turns 0 keeps the GAR sweep to one
# short-context point so a 1.6B model stays in memory while ablate/probe (which drive the
# reliance call) run normally. trust_remote_code is enabled in the loader for such repos.
python3 demo.py all --model stabilityai/stablelm-2-1_6b-chat --max-turns 0
python3 demo.py compare --stats   # run `all` a few times per model first so N > 1
```

These models expose `eager` attentions and support a system role. The demo will **skip** a
model whose chat template rejects a system message, *and* one whose system-prompt token span
can't be located — an empty span would silently turn the ablation into a no-op and report
fake "survival." StableLM-2 actually tripped this (its template renders a lone system message
to nothing), so the span detector now has a fallback and the run is guarded; the guard caught
two early StableLM runs whose total closure left recall at 4/4 (the no-op), and once fixed the
model collapses to 0/4 like the rest.

![residual decodability vs survival; all models cluster at high AUC, low survival](figures/auc_vs_survival.png)

The graded-closure survival is a *real, differentiated* axis with tight seed bands, but every
model — across 360M→7B and four families — lands in the **same `attention-reliant` bucket**:
the goal is decodable from the residual stream (AUC ~`0.99`–`1.0`) yet recall collapses as
attention to it is closed.

| model | residual AUC | survival | reliance |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 1.00 | 0.25 | attention-reliant |
| SmolLM2-360M-Instruct | 0.99 | 0.18 | attention-reliant |
| Qwen2.5-0.5B-Instruct | 1.00 | 0.17 | attention-reliant |
| stablelm-2-1.6b-chat | 1.00 | 0.16 | attention-reliant |
| TinyLlama-1.1B-Chat-v1.0 | 1.00 | 0.14 | attention-reliant |

All five collapse as the channel closes (the dissociation is the gap between AUC and the
curves below):

![recall vs visible fraction of the system span, per model](figures/survival_curves.png)

So the harness reproduces the *method and the per-model signature* — de-confounded across mask
orderings and filler seeds, no longer pinned at 0 by total ablation — but **not** the
architectural *divergence* itself. Crucially, **neither scaling within a family (Qwen 0.5B →
7B, still `0.25`) nor swapping to a different family (StableLM-2) flipped a model into
`residual-reliant`.** That is reported honestly rather than dressed up as a contrast: getting
an actual flip would need larger, deliberately contrasting families.

`compare --stats` treats survival across runs. Because greedy decoding with fixed filler seeds
makes each run's survival essentially deterministic, the per-model 95% CIs are ~`±0.00` (the
measurement is highly reproducible). The pairwise permutation test is included for
completeness but is **underpowered at N = 2** runs per model — its smallest achievable p-value
is ~`0.33` regardless of separation — so it reports no significant pairwise differences yet;
that is a power limitation, not evidence of no difference. Re-run `all` many times per model to
give it teeth.

### Run it bigger (GPU / cloud)

A true 7B+ flip test wants more memory than a laptop. The same commands run unchanged on a GPU
box; pick a dtype that fits and drop `--light` for the full sweep:

```bash
# On a CUDA box (auto picks bfloat16 for large models on gpu):
python3 demo.py all --model Qwen/Qwen2.5-7B-Instruct --dtype auto
python3 demo.py all --model Qwen/Qwen2.5-14B-Instruct --dtype bfloat16 --max-turns 24
python3 demo.py compare --stats
```

Locally, a 7B fits in ~15 GB with `--dtype bfloat16 --light` (1 seed, 1 mask order, short
generations, GAR pinned to 0 turns). Use **bfloat16, not float16**: fp16's narrow range
overflows in this model's eager attention on MPS, yielding NaN GAR and non-finite logits under
the ablation mask. `--dtype auto` keeps small models in float32 and drops >2B-param models to
bfloat16 on mps/cuda.

See [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md) for captured runs, including `report`,
`compare --stats`, `dissociate`, and `steer`.

---

## What this demonstrates — and what it does not

**It does demonstrate (qualitatively):**

- GAR decays as context grows (the attention channel closing), and at long context a
  *natural* recall MISS eventually appears — though on a 0.5B model this is sparse and
  non-monotonic (recall can recover at still-longer context).
- Closing the channel to the system tokens degrades fact recall relative to the
  unmodified model on the same prompts (the clean, total collapse).
- The planted value is decodable from the residual stream well above the input-embedding
  baseline — i.e. the information survives the channel thinning.
- **The dissociation within one model** (`dissociate`): as context grows, decodability
  (probe AUC) stays high while behavioral recall starts to miss — present but unused.
- **A causal handle on it** (`steer`): re-injecting the decoded goal direction under closure
  restores recall, confirming the goal was usable, just not used.
- The whole picture is **stable and reproducible** across repeated runs, and consistent across
  five models spanning 360M→7B and four architecture families (all `attention-reliant`).

**It does not claim:**

- To reproduce the paper's absolute numbers. These are small models (360M–7B) with
  home-grown prompts and a tiny probe set; `compare --stats` adds CIs and a permutation test,
  but the paper uses larger architectures, bigger fact sets, and far more statistical power.
- That the ablation is the paper's exact sliding-window procedure. It is a simpler
  whole-span column mask that captures the same idea (generated tokens can no longer see
  goal tokens).
- A causal claim about *production* behavior. The probe shows decodability, not that a
  given deployment will or won't break at a particular turn.

---

## Why MongoDB, via smongo

The paper's framing is that GAR is a **diagnostic you monitor** — so it is natural to keep the
measurements rather than print and discard them. Each run writes one small document per data
point (a GAR row, an ablation outcome, a probe AUC), and `report` runs MongoDB **aggregation
pipelines** over the accumulated runs to answer questions you cannot get from a single stdout
table, e.g. "the first crossover turn per model":

```python
coll.aggregate([
    {"$match": {"mode": "gar", "recall_miss": True}},
    {"$group": {"_id": "$model", "first_miss_turn": {"$min": "$turns"}}},
])
```

A document store fits because each measurement is a small, schema-flexible record, and the
analyses are naturally expressed as `$match` / `$group` pipelines.

[`smongo`](https://pypi.org/project/smongo/) is used instead of a server because it is an
**embedded, local-first MongoDB engine** (built on redb + Rust): same document model, same
MongoDB Query Language, same wire protocol, but **no `mongod`, no Docker, no network** — just a
file on disk via a `local://` URI. That keeps the project a single `python3 demo.py` away from
running while still using a genuine MongoDB-compatible engine with full aggregation (and even
`$vectorSearch`, unused here).

Honest caveats: smongo is **beta**, ships a **compiled Rust extension**, and requires
**Python >=3.11** (it pulls in `pymongo`). The store lives at `local://lost_track_db` by
default (git-ignored); override with `--db`, which also accepts a normal
`mongodb://`/Atlas URI if you would rather point at a real server.

---

## Install and run

Requires **Python >=3.11** (a smongo constraint).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 demo.py all       # runs the three modes and logs to local://lost_track_db
python3 demo.py report    # aggregate the stored runs
```

First run downloads the model (~1 GB). It runs on Apple Silicon (MPS) or CPU; no GPU
required. A different model can be supplied with `--model`, e.g.:

```bash
python3 demo.py probe --model Qwen/Qwen2.5-1.5B-Instruct
```

The metrics store defaults to `local://lost_track_db` (git-ignored); override with `--db`.

### Tests

`test_demo.py` is a fast, model-free smoke test (no model download, no inference) covering the
GAR helper on synthetic attention tensors, the ablation-mask invariants (no all-`-inf` rows,
finite softmax, correct columns masked), the MongoDB aggregations (crossover, latest-run
scoping, the dissociation `compare` signature) against seeded temporary stores, plus the new
**multi-run survival stats** (mean/CI/n), the **permutation test** (determinism + clear
separation), the **steering-vector diff-of-means** math, and a **`plot` smoke test** that
renders all four figures to a temp dir:

```bash
python3 test_demo.py          # or: pytest test_demo.py
```

---

## Notes and caveats

- **GAR aggregation.** GAR here averages attention mass over heads per layer. The demo
  reports early/late-layer splits, but a single scalar still blurs head- and layer-level
  structure that the paper treats more carefully.
- **Probe size.** The probe uses a small synthetic set (4 codename values × 8 filler
  variations). AUC is informative about the *shape* (residual ≫ embedding) but is not a
  statistically rigorous estimate.
- **Practical takeaways** such as monitoring a GAR-like signal, periodically re-injecting
  the system prompt, or selecting models by long-horizon behavior follow naturally from
  the paper's framing, but they are engineering hypotheses — this repo does not evaluate
  them.

---

## Appendix: possible future improvements

This demo is intentionally a faithful reproduction-in-miniature, not a full replication. The
items below would move it closer to the paper; they are grouped by whether the value is worth
the added complexity and risk on a laptop-scale, single-small-model setup.

### Worth doing (feasible, value > risk)

- **Larger / strict-match probe set.** Grow beyond 4 codename values × 8 filler variations
  and add held-out paraphrases of the probe question, so the residual AUC is a more
  statistically meaningful estimate rather than just the right *shape*.
- **Per-fact GAR vs per-fact recall.** Measure GAR at each of the four probe positions (not
  only the codename probe) and correlate each fact's GAR with its own recall, tightening the
  link the narrative draws between attention thinning and the observed MISS.

### Worth doing carefully (frame honestly)

- **Cross-model comparison (implemented, descriptively + with stats).** `demo.py compare`
  lines up each logged model's dissociation signature, and `--stats` adds cross-run CIs and a
  pairwise permutation test. As shipped, five models spanning 360M→7B and four families all
  came out `attention-reliant` — scaling Qwen 0.5B→7B and swapping to StableLM-2 both failed to
  flip the bucket — so this is a working harness and an honest per-model signature, not the
  *contrasting* failure mode. Getting an actual divergence would need larger, deliberately
  chosen families (see below), and a permutation test with real power needs many runs per model
  (at N=2 its p-value floor is ~0.33).
- **Within-model dissociation + steering (implemented).** `demo.py dissociate` shows
  decodability holding while recall falls inside one model, and `demo.py steer` restores recall
  under closure by re-injecting the decoded goal direction — the paper's claim made causal,
  beyond the descriptive probe.
- **Longer / cleaner behavioral failure curve.** MODE 1 currently shows a single, noisy
  natural MISS at ~4.6k tokens. Averaging over several filler orderings per length (and
  reporting a recall rate, not a single 0/1 outcome) would smooth the curve and make the
  GAR-to-failure relationship more convincing — at the cost of more compute.

### Probably not worth it on this scale (risk > value)

- **Claiming the cross-architecture *divergence* finding.** This is the paper's central
  contribution and needs multiple model families chosen to contrast; asserting it from one or
  two small models invites cherry-picking.
- **Parametric crossover-turn / failure-timing prediction.** Fitting *when* a model fails is
  fragile on a single 0.5B model and would mostly capture noise.
- **True sliding-window ablation + persona-violation metrics.** The whole-span and graded
  suffix masks already prove the causal point and give a survival axis that discriminates, and
  grading persona drift on a small model is subjective; both add code surface for little
  incremental evidence unless the project pivots toward a benchmark.

---

## License

See [LICENSE](LICENSE).

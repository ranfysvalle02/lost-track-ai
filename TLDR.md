# TLDR

A laptop-runnable demo that reproduces the *shape* of **"When Attention Closes: How LLMs Lose
the Thread in Multi-Turn Interaction"** (Dongre et al., [arXiv:2605.12922](https://arxiv.org/abs/2605.12922))
by reading model internals (attention matrices + hidden states) on small models — default
`Qwen/Qwen2.5-0.5B-Instruct`, MPS/CPU, no GPU required.

## The one idea

LLMs drift from their system prompt over long chats not because the goal is *gone*, but
because it becomes **present but unused**. A goal reaches later tokens by two channels:

- **Attention channel** — later tokens attend back to the system-prompt (goal) tokens.
- **Residual stream** — a latent trace of the goal carried forward in hidden states.

As context grows, the **attention channel closes** (attention onto goal tokens thins) while
the **residual trace lingers**. So the goal stays *decodable* from hidden states yet stops
*driving behavior*. Apparent "forgetting" is a routing failure, not information loss.

## Three instruments (one-to-one with the paper) + two that go further

| mode | what it shows | headline number |
|---|---|---|
| `gar` | **Goal Accessibility Ratio** decays as context grows | GAR `0.58 → 0.33` over `103 → 5,763` tokens; first natural recall MISS at ~4.6k tokens |
| `ablate` | force-closing the channel collapses recall | total closure: recall `4/4 → 0/4` |
| `probe` | the goal survives in the residual stream | residual AUC `0.99–1.0` vs embedding baseline `0.50` (chance) |
| `dissociate` | **the headline, in one model**: AUC stays high while recall falls | AUC `0.77–1.0` while recall drops `4/4 → 3/4` as GAR falls |
| `steer` | causal proof: re-inject the goal direction, recall returns | `MISS → "Halcyon."` under total closure once the diff-of-means vector is added |

`report` / `compare` / `plot` read the stored runs (no model load) to aggregate and chart.

## Why "decodable" ≠ "used" (the crux)

- **Decodable** = an *external* linear probe finds a goal-encoding direction in the hidden state.
- **Used** = the model's *own* fixed circuit routes that goal into its next-token logits.

These are different computations. When attention to the goal tokens thins, the model's own
readout no longer conditions on the lingering trace — even though a probe still recovers it.
`dissociate` watches the two curves separate on the *same* context; `steer` upgrades "present"
to "present and usable" by re-injecting the direction at the decision point and restoring recall.

> Note: `dissociate` measures decodability under **natural attention decay** (more filler), not
> the ablation mask. A hard column mask severs the path to the goal, so AUC would collapse *with*
> recall — leaving nothing to dissociate.

## Honest negative: no cross-architecture *flip*

`compare --stats` lines up five models (360M→7B, four families) on residual AUC vs
**graded-closure survival** (order- and seed-averaged recall as the system prompt is hidden).
The survival axis is real and differentiated (`0.14–0.25`), but **every model buckets as
`attention-reliant`** — decodable (AUC `~0.99–1.0`) yet recall collapses under closure.

- Scaling within a family (Qwen `0.5B → 7B`): survival only `0.17 → 0.25`.
- Switching family (StableLM-2): `0.16`.

Neither flipped a model into `residual-reliant`. The paper's architectural *divergence* does
**not** appear at this scale — reported as a clean negative, not massaged into a contrast.

## What it does / doesn't claim

- **Does** (qualitatively): GAR decay, causal recall collapse under closure, residual
  decodability ≫ embedding baseline, the within-model dissociation, and a causal steering handle
  — stable across repeated runs and five models.
- **Doesn't**: match the paper's absolute numbers, use its exact sliding-window ablation (uses a
  whole-span / graded column mask), claim the cross-architecture divergence, or make
  production-behavior or failure-timing predictions.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate   # Python >= 3.11 (smongo constraint)
pip install -r requirements.txt

python3 demo.py all          # gar + ablate + probe, logged to local://lost_track_db
python3 demo.py dissociate   # decodable-but-unused, one model
python3 demo.py steer        # re-inject the goal, recall returns
python3 demo.py report       # MongoDB aggregations over stored runs (no model load)
python3 demo.py compare --stats   # cross-architecture + CIs + permutation test
python3 demo.py plot --all-runs   # render figures/ from the store
python3 test_demo.py         # fast, model-free smoke test (no download/inference)
```

## Why MongoDB (via smongo)

The paper frames GAR as a **diagnostic you monitor**, so each measurement is logged as one
small document and `report`/`compare` run real MongoDB `$match`/`$group` aggregation pipelines
over the accumulated runs. [`smongo`](https://pypi.org/project/smongo/) is an embedded,
local-first MongoDB engine (redb + Rust): same query language and wire protocol, but no
`mongod`, no Docker, no network — a single file on disk via `local://`. (Beta; ships a compiled
Rust extension; needs Python ≥ 3.11.)

## Caveats

- 0.5B natural failures are **sparse and non-monotonic** (recall can recover at longer context);
  the *shape* is the paper's, not the precision.
- GAR averages attention over heads, blurring head/layer structure the paper treats carefully.
- The probe set is small (4 codenames × 8 fillers = 32 samples) — informative about shape, not a
  rigorous estimate. The permutation test is underpowered at N=2 runs (p-floor ~`0.33`).

See [README.md](README.md), [PAPER.md](PAPER.md) (instrument-by-instrument mapping), and
[SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md) (captured runs + figures).

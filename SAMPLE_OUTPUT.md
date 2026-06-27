# Sample output

A real run of `python3 demo.py all` on `Qwen/Qwen2.5-0.5B-Instruct` (MPS, float32).
Absolute numbers will vary by machine, transformers version, and model — the point is the
*shape* of each result. Framework noise (download progress bars, HF token notice) is
omitted below.

```text
loading Qwen/Qwen2.5-0.5B-Instruct on mps ...

========================================================================
  MODE 1: GAR decay — attention thins, then recall finally breaks
========================================================================
turns | ctx tok |  GAR all |   early |    late | recall
------------------------------------------------------------------------
    0 |     103 |   0.5821 |  0.5077 |  0.6874 |    4/4
    8 |     386 |   0.3783 |  0.2556 |  0.5244 |    4/4
   24 |     952 |   0.3660 |  0.2246 |  0.5136 |    4/4
   56 |    2084 |   0.3526 |  0.2143 |  0.4899 |    4/4
   96 |    3499 |   0.3363 |  0.1998 |  0.4617 |    4/4
  128 |    4631 |   0.3383 |  0.2006 |  0.4582 |    3/4
  160 |    5763 |   0.3316 |  0.1942 |  0.4507 |    4/4
------------------------------------------------------------------------
GAR trends down as context grows: the attention channel is closing.
First recall MISS at 128 turns (~4631 tokens), GAR 0.3383: the model
drops a planted fact once attention to the system prompt has thinned.
On a 0.5B model natural failures are sparse and noisy (recall can recover
at longer context) — MODE 2 forces the clean *causal* collapse via ablation.
(early = first third of layers, late = last third.)


========================================================================
  MODE 2: Ablation — force-close attention to system tokens, recall collapses
========================================================================
                fact |   normal |  ablated
------------------------------------------------------------------------
    project codename |       OK |     MISS
        lead auditor |       OK |     MISS
compliance framework |       OK |     MISS
      launch quarter |       OK |     MISS
------------------------------------------------------------------------
recall: normal 4/4 (100%)  ->  ablated 0/4 (0%)
Blinding the model to its own system prompt collapses fact recall.


========================================================================
  MODE 3: Residual probe — the planted goal survives in the hidden states
========================================================================
  samples=32  classes=4 (codename values)  layers=24
------------------------------------------------------------------------
  residual stream (layer  2)      : AUC 0.500
  residual stream (layer 12)      : AUC 0.990
  residual stream (layer 22)      : AUC 1.000
  input embeddings  (layer  0)      : AUC 0.500  (chance — last-token input is identical across classes)
------------------------------------------------------------------------
The planted codename is decodable from the residual stream far above the
embedding baseline: the value survives in the hidden state even as the
attention channel to the system prompt thins.

logged 15 metric docs to local://lost_track_db (run_id b52653f7) — run `python3 demo.py report` to aggregate across runs.
```

## `report` mode (MongoDB aggregations over stored runs)

Every run logs its measurements to an embedded smongo store; `python3 demo.py report` runs
aggregation pipelines over them. By default it scopes to the **latest run per model**, so
re-running `all` never inflates or blends the numbers (ablation is shown as a per-fact
rate, `ok/facts`):

```text
========================================================================
  REPORT: MongoDB aggregations over stored runs
========================================================================
  scope: latest run per model (use --all-runs for full history)
  15 documents across 1 model(s): Qwen/Qwen2.5-0.5B-Instruct

  GAR decay (max -> min) and context reached:
                           model |  max GAR |  min GAR | max turns
      Qwen/Qwen2.5-0.5B-Instruct |   0.5821 |   0.3316 |       160

  First recall MISS (crossover turn):
                           model | first miss turn | GAR there
      Qwen/Qwen2.5-0.5B-Instruct |             128 |    0.3383

  Ablation recall (rate over facts, normal vs ablated):
                           model |   normal |  ablated | facts
      Qwen/Qwen2.5-0.5B-Instruct |     1.00 |     0.00 |     4

  Best probe AUC (residual vs embedding baseline):
                           model |      repr | best AUC
      Qwen/Qwen2.5-0.5B-Instruct | embedding |    0.500
      Qwen/Qwen2.5-0.5B-Instruct |  residual |    1.000
```

Run `demo.py all` a second time and the default report still reads `facts | 4` (latest run
only) — the numbers don't double. `python3 demo.py report --all-runs` aggregates the full
history instead (here, two runs -> `facts | 8`), while the rates and min/max GAR stay
stable; `--run <run_id>` scopes to a single run. Re-running with `--model <other>` makes the
report aggregate across models.

## How to read it

- **GAR decay, then a behavioral MISS.** `GAR all` drops from 0.58 to ~0.33 as the
  context grows from 103 to 5,763 tokens — the attention channel onto the system prompt is
  thinning, and the `early`/`late` split shows the early layers thin faster. Recall stays
  4/4 until the first MISS at 128 turns (~4,631 tokens), where the model drops the launch
  quarter. The failure is genuine but sparse and non-monotonic (recall is 4/4 again at
  5,763 tokens): on a 0.5B model natural attention decay rarely produces a clean collapse,
  which is why MODE 2 induces it causally. GAR here is measured memory-safely at the final
  token (`Model.gar_last_token`), so the sweep can reach multi-thousand-token context
  without materializing O(L^2) attention across all layers.
- **Ablation.** With normal attention the model recalls every fact (4/4). Closing the
  channel from generated tokens to the system span drops recall to 0/4 — and the run
  completes with no NaN (the finite-logits guard in `generate_ablated` passed), so the
  collapse is the manipulation, not a numerical artifact.
- **Residual probe.** The planted codename is undecodable at layer 2 (AUC 0.500), then
  becomes almost perfectly decodable by layers 12 and 22 (0.990 / 1.000), while the
  input-embedding baseline stays at chance (0.500) because the final-token input is
  identical across classes. That gap is the paper's point reproduced in miniature: the
  goal survives in the residual stream, and the layer where it emerges is well above the
  input.

The model-free smoke test (`python3 test_demo.py`) covers the GAR math, the ablation-mask
invariants, the MongoDB crossover aggregation and latest-run scoping (against seeded temp
stores), and that logging is best-effort — all without any download or inference.
```text
[warn] metric not logged: disk on fire
ok  test_ablation_mask_has_no_all_inf_rows
ok  test_ablation_mask_shape_and_columns
ok  test_crossover_by_model_aggregation
ok  test_gar_per_layer_matches_hand_calc
ok  test_gar_tracks_span_mass
ok  test_log_metric_is_best_effort
ok  test_report_scope_latest_run

7 passed
```

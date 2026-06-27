#!/usr/bin/env python3
"""Fast, model-free smoke tests for demo.py.

These never download or run the model — they only exercise the pure-tensor helpers and the
embedded-MongoDB aggregations:

  * gar_per_layer / gar_from_attentions  — Goal Accessibility Ratio math
  * build_ablation_mask                  — the attention mask that closes the channel
                                           from generated tokens to goal tokens
  * crossover_by_model                   — MongoDB aggregation over logged metrics (smongo)

Run with either:

  python3 test_demo.py
  pytest test_demo.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone

import torch

from demo import (
    GAR_SCHEDULE,
    NEG,
    build_ablation_mask,
    classify_reliance,
    compare_by_model,
    crossover_by_model,
    gar_from_attentions,
    gar_per_layer,
    gar_schedule,
    latest_run_ids,
    log_metric,
    open_metrics,
)


def _attn(values: list[list[float]]) -> torch.Tensor:
    """Build a (batch=1, heads, q, k) attention tensor for one layer from a per-head list
    of rows. Each `values[h]` is the attention row at the query position we probe."""
    heads = len(values)
    k = len(values[0])
    a = torch.zeros(1, heads, 1, k)
    for h, row in enumerate(values):
        a[0, h, 0] = torch.tensor(row)
    return a


def test_gar_per_layer_matches_hand_calc() -> None:
    # query position 0; system span = first 2 of 4 key positions.
    # layer 0: head0 mass on span = 0.5+0.1=0.6, head1 = 0.2+0.2=0.4 -> mean 0.5
    # layer 1: head0 = 0.1+0.1=0.2, head1 = 0.4+0.4=0.8 -> mean 0.5
    layer0 = _attn([[0.5, 0.1, 0.3, 0.1], [0.2, 0.2, 0.3, 0.3]])
    layer1 = _attn([[0.1, 0.1, 0.4, 0.4], [0.4, 0.4, 0.1, 0.1]])
    attentions = [layer0, layer1]
    span = slice(0, 2)

    per_layer = gar_per_layer(attentions, query_pos=0, span=span)
    assert per_layer == [0.5, 0.5], per_layer

    gar = gar_from_attentions(attentions, query_pos=0, span=span)
    assert abs(gar - 0.5) < 1e-6, gar


def test_gar_tracks_span_mass() -> None:
    # All mass inside the span -> GAR == 1.0; all mass outside -> GAR == 0.0.
    inside = [_attn([[0.5, 0.5, 0.0, 0.0]])]
    outside = [_attn([[0.0, 0.0, 0.5, 0.5]])]
    span = slice(0, 2)
    assert abs(gar_from_attentions(inside, 0, span) - 1.0) < 1e-6
    assert abs(gar_from_attentions(outside, 0, span) - 0.0) < 1e-6


def test_ablation_mask_shape_and_columns() -> None:
    L, sys_stop = 6, 3
    sys_span = slice(0, sys_stop)
    mask = build_ablation_mask(L, sys_span)
    assert mask.shape == (1, 1, L, L)

    m = mask[0, 0]
    # System rows keep their causal self-attention to the system columns.
    for i in range(sys_stop):
        assert m[i, i].item() != NEG, f"system row {i} lost its diagonal"
    # Post-system rows are blinded to every system column...
    for i in range(sys_stop, L):
        for j in range(sys_stop):
            assert m[i, j].item() == NEG, f"row {i} should not see system col {j}"
        # ...but still see their own (causal) diagonal.
        assert m[i, i].item() != NEG, f"row {i} lost its diagonal"


def test_ablation_mask_has_no_all_inf_rows() -> None:
    # The whole point of the fix: no fully -inf row -> softmax produces no NaNs.
    L, sys_span = 8, slice(0, 4)
    mask = build_ablation_mask(L, sys_span)[0, 0]
    attn = torch.softmax(mask, dim=-1)
    assert torch.isfinite(attn).all(), "softmax produced non-finite values (all -inf row)"
    # Each row's probabilities should sum to 1 (no degenerate NaN row).
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums


def test_crossover_by_model_aggregation() -> None:
    # Seed an embedded smongo store with synthetic GAR metrics for two models and check the
    # MongoDB aggregation reports the correct first-MISS turn per model.
    tmp = tempfile.mkdtemp()
    try:
        coll = open_metrics("local://" + os.path.join(tmp, "db"))
        docs = [
            {"model": "A", "mode": "gar", "turns": 8, "gar_all": 0.40, "recall_miss": False},
            {"model": "A", "mode": "gar", "turns": 64, "gar_all": 0.35, "recall_miss": False},
            {"model": "A", "mode": "gar", "turns": 128, "gar_all": 0.33, "recall_miss": True},
            {"model": "A", "mode": "gar", "turns": 160, "gar_all": 0.31, "recall_miss": True},
            {"model": "B", "mode": "gar", "turns": 96, "gar_all": 0.30, "recall_miss": True},
            # A non-gar doc that must be ignored by the pipeline's $match.
            {"model": "B", "mode": "ablate", "fact": "x", "normal_ok": 1, "ablated_ok": 0},
        ]
        coll.insert_many(docs)

        rows = {r["_id"]: r for r in crossover_by_model(coll)}
        assert set(rows) == {"A", "B"}, rows
        assert rows["A"]["first_miss_turn"] == 128, rows["A"]
        assert rows["B"]["first_miss_turn"] == 96, rows["B"]
        assert abs(rows["A"]["lowest_gar_seen"] - 0.31) < 1e-9, rows["A"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_report_scope_latest_run() -> None:
    # Two runs for the same model: an older clean run (no MISS) and a newer run with a MISS.
    # Scoping to the latest run per model must surface only the newer run's data.
    tmp = tempfile.mkdtemp()
    try:
        coll = open_metrics("local://" + os.path.join(tmp, "db"))
        old_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        new_ts = old_ts + timedelta(hours=1)
        coll.insert_many([
            # Older run: recall held the whole way (no MISS).
            {"model": "A", "run_id": "old", "ts": old_ts, "mode": "gar",
             "turns": 64, "gar_all": 0.40, "recall_miss": False},
            {"model": "A", "run_id": "old", "ts": old_ts, "mode": "gar",
             "turns": 128, "gar_all": 0.38, "recall_miss": False},
            # Newer run: a genuine MISS at turn 96.
            {"model": "A", "run_id": "new", "ts": new_ts, "mode": "gar",
             "turns": 64, "gar_all": 0.36, "recall_miss": False},
            {"model": "A", "run_id": "new", "ts": new_ts, "mode": "gar",
             "turns": 96, "gar_all": 0.34, "recall_miss": True},
        ])

        assert latest_run_ids(coll) == ["new"], latest_run_ids(coll)

        match = {"run_id": {"$in": latest_run_ids(coll)}}
        rows = {r["_id"]: r for r in crossover_by_model(coll, match)}
        assert set(rows) == {"A"}, rows
        assert rows["A"]["first_miss_turn"] == 96, rows["A"]

        # Full history still sees the same single MISS (the older run had none).
        all_rows = {r["_id"]: r for r in crossover_by_model(coll)}
        assert all_rows["A"]["first_miss_turn"] == 96, all_rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_log_metric_is_best_effort() -> None:
    # A telemetry write failure must never propagate out of log_metric.
    class _Boom:
        def insert_one(self, _doc):
            raise RuntimeError("disk on fire")

    log_metric(_Boom(), {"run_id": "r"}, {"mode": "gar", "turns": 8})  # must not raise
    log_metric(None, {"run_id": "r"}, {"mode": "gar"})  # None coll is a no-op


def test_gar_schedule_caps_by_model_size() -> None:
    # Small models get the full sweep; larger ones are capped so eager-attention prefill
    # stays in memory. An explicit max_turns always wins.
    assert gar_schedule(0.5e9) == GAR_SCHEDULE
    assert gar_schedule(1.1e9)[-1] == 56, gar_schedule(1.1e9)
    assert gar_schedule(1.7e9)[-1] == 24, gar_schedule(1.7e9)
    assert gar_schedule(1.7e9, max_turns=160) == GAR_SCHEDULE
    assert gar_schedule(0.5e9, max_turns=8) == (0, 8)
    # Never empty, even for an absurdly low cap.
    assert gar_schedule(0.5e9, max_turns=0) == (0,)


def test_classify_reliance_buckets() -> None:
    # Goal not decodable -> weak-encoding regardless of survival.
    assert classify_reliance(0.55, 1.0) == "weak-encoding"
    assert classify_reliance(None, None) == "weak-encoding"
    # Decodable + recall holds under closure -> residual-reliant (robust).
    assert classify_reliance(0.99, 1.0) == "residual-reliant"
    assert classify_reliance(0.99, 0.50) == "residual-reliant"
    # Decodable but no closure evidence -> residual-reliant by default.
    assert classify_reliance(0.99, None) == "residual-reliant"
    # Decodable but recall collapses under closure -> the dissociation.
    assert classify_reliance(0.99, 0.0) == "attention-reliant"
    assert classify_reliance(0.99, 0.49) == "attention-reliant"


def test_compare_dissociation() -> None:
    # Two synthetic models with identical residual decodability but opposite behavior under
    # attention closure: R keeps recall (residual-reliant), F collapses (attention-reliant).
    tmp = tempfile.mkdtemp()
    try:
        coll = open_metrics("local://" + os.path.join(tmp, "db"))
        docs = []
        for model, ablated_ok in (("R", 1), ("F", 0)):
            # Probe: residual highly decodable, embedding at chance.
            docs.append({"model": model, "mode": "probe", "repr": "residual", "auc": 0.99})
            docs.append({"model": model, "mode": "probe", "repr": "embedding", "auc": 0.50})
            # Ablation: both recall all 4 facts normally; only R survives closure.
            for fact in ("a", "b", "c", "d"):
                docs.append({"model": model, "mode": "ablate", "fact": fact,
                             "normal_ok": 1, "ablated_ok": ablated_ok})
        coll.insert_many(docs)

        rows = {r["model"]: r for r in compare_by_model(coll)}
        assert set(rows) == {"R", "F"}, rows
        assert abs(rows["R"]["residual_auc"] - 0.99) < 1e-9
        assert rows["R"]["survival"] == 1.0, rows["R"]
        assert rows["F"]["survival"] == 0.0, rows["F"]
        assert rows["R"]["reliance"] == "residual-reliant", rows["R"]
        assert rows["F"]["reliance"] == "attention-reliant", rows["F"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    _run_all()

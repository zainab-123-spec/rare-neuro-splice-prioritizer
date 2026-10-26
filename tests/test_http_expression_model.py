"""HTTP caching, expression math, cross-validation, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rnsp.cli import build_parser
from rnsp.config import DISCLAIMER, GTEX_BRAIN_TISSUES, MODEL_FEATURES_FULL
from rnsp.expression import (
    brain_expression_features,
    expression_query_symbols,
    parse_hpa,
    tau_specificity,
)
from rnsp.http import HttpClient, HttpError, RateLimiter
from rnsp.model import (
    _iter_splits,
    balanced_weights,
    crossval_oof,
    dinucleotide_rule_scores,
    distance_rule_scores,
    evaluate,
    subset_mask,
    summarize_predictions,
)
from rnsp.report import render_benchmark_html, render_gene_html


class _Response:
    def __init__(self, status: int, content: bytes) -> None:
        self.status_code = status
        self.content = content
        self.url = "https://example.test/resource"
        self.headers: dict[str, str] = {}


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.headers: dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls += 1
        if not self._responses:
            raise AssertionError("session received an unexpected call")
        return self._responses.pop(0)


def test_rate_limiter_waits_only_when_the_interval_has_not_elapsed() -> None:
    clock = {"now": 100.0}
    slept: list[float] = []

    def sleep(delay: float) -> None:
        slept.append(delay)
        clock["now"] += delay

    limiter = RateLimiter(0.5, clock=lambda: clock["now"], sleep=sleep)
    limiter.wait()
    limiter.wait()
    assert slept == [0.5]
    limiter.wait()
    assert slept == [0.5, 0.5]


def test_http_retries_503_then_uses_the_cache(tmp_path: Path) -> None:
    session = _Session(
        [
            _Response(503, b"busy"),
            _Response(200, b'{"ok": 1}'),
        ]
    )
    client = HttpClient(
        tmp_path,
        min_interval=0,
        max_retries=3,
        session=session,  # type: ignore[arg-type]
        sleep=lambda _delay: None,
    )
    assert client.get_json("https://example.test/resource") == {"ok": 1}
    assert client.get_json("https://example.test/resource") == {"ok": 1}
    assert session.calls == 2


def test_http_does_not_retry_a_client_error(tmp_path: Path) -> None:
    session = _Session([_Response(404, b"missing")])
    client = HttpClient(
        tmp_path,
        min_interval=0,
        max_retries=4,
        session=session,  # type: ignore[arg-type]
        sleep=lambda _delay: None,
    )
    with pytest.raises(HttpError) as caught:
        client.get_json("https://example.test/missing")
    assert caught.value.status == 404
    assert session.calls == 1


def test_gtex_brain_specificity_and_tau() -> None:
    records = [
        {"tissueSiteDetailId": "Brain_Cortex", "median": 10.0},
        {"tissueSiteDetailId": "Brain_Hippocampus", "median": 6.0},
        {"tissueSiteDetailId": "Brain_Cerebellum", "median": 8.0},
        {"tissueSiteDetailId": "Liver", "median": 1.0},
        {"tissueSiteDetailId": "Lung", "median": 1.0},
    ]
    summary = brain_expression_features(records)
    assert summary["gtex_cortex_tpm"] == 10
    assert summary["gtex_hippocampus_tpm"] == 6
    assert summary["gtex_cerebellum_tpm"] == 8
    assert summary["gtex_brain_specificity"] > 1
    assert 0 < tau_specificity([10, 0, 0, 0]) <= 1
    assert tau_specificity([5, 5, 5]) == pytest.approx(0)


def test_hpa_parser_reads_brain_ntpm() -> None:
    parsed = parse_hpa(
        {
            "Gene": "SCN1A",
            "Ensembl": "ENSG00000144285",
            "RNA tissue specificity": "Tissue enhanced",
            "RNA brain regional specificity": "Low region specificity",
            "RNA tissue specific nTPM": {"brain": "7.9", "retina": "3.1"},
        }
    )
    assert parsed["hpa_gene"] == "SCN1A"
    assert parsed["hpa_brain_ntpm"] == pytest.approx(7.9)
    assert parsed["hpa_rna_tissue_specificity"] == "Tissue enhanced"


def test_gtex_query_tries_the_historical_gba_symbol() -> None:
    assert expression_query_symbols("GBA1", ("GBA",)) == ("GBA1", "GBA")
    assert expression_query_symbols("SCN1A", ()) == ("SCN1A",)


def test_gtex_brain_panel_includes_cortex_hippocampus_cerebellum() -> None:
    assert "Brain_Cortex" in GTEX_BRAIN_TISSUES
    assert "Brain_Hippocampus" in GTEX_BRAIN_TISSUES
    assert "Brain_Cerebellum" in GTEX_BRAIN_TISSUES


def _synthetic(n_genes: int = 5, per_class: int = 8) -> pd.DataFrame:
    rows = []
    for gene_index in range(n_genes):
        gene = f"G{gene_index}"
        for label, distance, region, flag, delta in (
            (1, 1, "canonical_dinucleotide", 1, -8.0),
            (0, 70, "deep_gt_50", 0, 0.0),
            (1, 5, "hard_3_50", 0, -6.0),
            (0, 40, "hard_3_50", 0, 0.2),
        ):
            for copy in range(per_class // 2):
                row = {name: 0.0 for name in MODEL_FEATURES_FULL}
                row.update(
                    {
                        "gene": gene,
                        "label": label,
                        "chrom": "2",
                        "pos": 1000 * gene_index + copy + label * 100,
                        "ref": "A",
                        "alt": "G",
                        "region_class": region,
                        "criteria_provided": 1,
                        "abs_dist_to_splice": distance,
                        "disrupts_canonical_dinucleotide": flag,
                        "donor_maxent_delta": delta,
                        "clinical_significance": "Pathogenic" if label else "Benign",
                        "hgvs_like_offset": distance if label else -distance,
                        "nearest_exon_rank": 0,
                        "nearest_is_donor": 1,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def test_group_kfold_holds_out_whole_genes() -> None:
    frame = _synthetic()
    groups = frame["gene"].to_numpy()
    matrix = frame[["abs_dist_to_splice"]].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    for train, test in _iter_splits(matrix, y, groups, "group_kfold_gene", 5):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        assert len(test) > 0


def test_stratified_split_keeps_both_classes() -> None:
    frame = _synthetic()
    y = frame["label"].to_numpy()
    matrix = frame[["abs_dist_to_splice"]].to_numpy(dtype=float)
    groups = frame["gene"].to_numpy()
    for _train, test in _iter_splits(matrix, y, groups, "stratified_kfold", 5):
        assert set(y[test].tolist()) == {0, 1}


def test_distance_and_dinucleotide_rules() -> None:
    scores = distance_rule_scores(np.array([1, 10, 50]))
    assert scores[0] > scores[1] > scores[2]
    assert list(dinucleotide_rule_scores(np.array([1, 0, 1]))) == [1.0, 0.0, 1.0]
    frame = _synthetic()
    mask = subset_mask(frame, "hard_3_50")
    assert set(frame.loc[mask, "region_class"]) == {"hard_3_50"}
    assert int(frame.loc[mask, "disrupts_canonical_dinucleotide"].sum()) == 0


def test_balanced_weights_give_each_class_equal_mass() -> None:
    labels = np.array([1, 1, 0, 0, 0, 0])
    weights = balanced_weights(labels)
    assert weights[labels == 1].sum() == pytest.approx(weights[labels == 0].sum())


def test_bootstrap_interval_contains_the_point_estimate() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.05, 0.3, 0.9, 0.8, 0.7, 0.95, 0.2, 0.85])
    summary = summarize_predictions(y, scores, n_bootstrap=40, min_per_class=4)
    assert summary is not None
    low, high = summary["roc_auc_ci95"]
    assert low <= summary["roc_auc"] <= high
    assert 0 <= summary["brier"] <= 1


def test_gene_held_out_model_beats_chance_when_distance_separates_labels() -> None:
    frame = _synthetic()
    oof = crossval_oof(
        frame,
        ["abs_dist_to_splice"],
        "logistic_regression",
        "group_kfold_gene",
        n_splits=5,
    )
    summary = summarize_predictions(frame["label"].to_numpy(), oof, n_bootstrap=10)
    assert summary is not None
    assert summary["roc_auc"] > 0.8


def test_evaluate_records_only_finite_metrics(tmp_path: Path) -> None:
    frame = _synthetic()
    metrics, scored, _estimator = evaluate(
        frame,
        n_splits=5,
        n_bootstrap=15,
        models=("logistic_regression", "hist_gradient_boosting"),
        feature_sets=("full", "motif"),
    )
    assert metrics["n_variants"] == len(frame)
    headline = [
        row
        for row in metrics["evaluations"]
        if row.get("model") == "logistic_regression"
        and row.get("split") == "group_kfold_gene"
        and row.get("subset") == "all"
        and row.get("feature_set") == "full"
        and row.get("protocol") == "train_all_eval_subset"
    ]
    assert len(headline) == 1
    assert headline[0]["roc_auc"] > 0.8
    assert "oof_lr_group_full" in scored.columns
    assert scored["deployment_score"].between(0, 1).all()
    document = render_benchmark_html(
        metrics, {"drop_reasons": {"reference_mismatch": 3}, "n_variants": 1}
    )
    assert DISCLAIMER in document
    assert "0.000" not in json.dumps(headline[0]["roc_auc"]) or headline[0]["roc_auc"] > 0
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_gene_report_escapes_markup_and_states_the_disclaimer() -> None:
    frame = _synthetic(n_genes=1, per_class=4)
    frame["gene"] = "<script>"
    frame["deployment_score"] = 0.5
    structure = {
        "disease": "test",
        "canonical_transcript": "ENST1",
        "strand": 1,
        "chrom": "2",
        "start": 1,
        "end": 10,
        "symbol": "<script>",
    }
    document = render_gene_html("<script>", frame, structure)
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert DISCLAIMER in document


def test_gene_panel_lists_the_thirty_neurological_genes() -> None:
    from rnsp.config import GENE_PANEL

    symbols = [gene.symbol for gene in GENE_PANEL]
    assert len(symbols) == 30
    assert len(set(symbols)) == 30
    for required in ("SCN1A", "MECP2", "DMD", "SMN1", "GBA1", "CFTR", "CACNA1A"):
        assert required in symbols


def test_cli_parses_analyze_gene() -> None:
    parser = build_parser()
    args = parser.parse_args(["analyze", "SCN1A", "--genes", "SCN1A,MECP2"])
    assert args.command == "analyze"
    assert args.gene == "SCN1A"
    build = parser.parse_args(["build"])
    assert build.command == "build"
    assert build.genes == ""

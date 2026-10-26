"""Publication figures for the benchmark and for one-gene architecture maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from rnsp.config import GTEX_BRAIN_TISSUES
from rnsp.model import format_metric, metric_lookup

PATHOGENIC = "#b2182b"
BENIGN = "#2166ac"
MODEL_COLORS = {
    "logistic_regression": "#1b4f72",
    "hist_gradient_boosting": "#b9770e",
    "gradient_boosting": "#0e6655",
    "distance_rule": "#666666",
    "dinucleotide_rule": "#111111",
    "distance_logistic": "#7f8c8d",
}


def apply_style() -> None:
    """Set a quiet Matplotlib style used by every saved figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
        }
    )


def write_figures(
    scored: pd.DataFrame,
    metrics: dict[str, Any],
    gtex: pd.DataFrame | None,
    structures: dict[str, Any],
    out_dir: Path,
) -> list[Path]:
    """Write the benchmark figure set and return the file paths."""
    apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        _roc_figure(scored, out_dir / "roc_curves.png"),
        _pr_figure(scored, out_dir / "pr_curves.png"),
        _metric_bars(metrics, out_dir / "metrics_comparison.png"),
        _hard_subset_bars(metrics, out_dir / "hard_subset_metrics.png"),
        _calibration(scored, out_dir / "calibration.png"),
        _coefficients(metrics, out_dir / "logistic_coefficients.png"),
        _importance(metrics, out_dir / "permutation_importance.png"),
        _distance_hist(scored, out_dir / "distance_by_label.png"),
        _cohort(scored, out_dir / "cohort_by_gene.png"),
    ]
    if gtex is not None and not gtex.empty:
        written.append(_expression_heatmap(gtex, scored, out_dir / "brain_expression.png"))
    highlight = "SCN1A" if "SCN1A" in structures else next(iter(structures))
    if highlight in set(scored["gene"]) and highlight in structures:
        written.append(
            plot_gene_architecture(
                scored[scored["gene"] == highlight],
                structures[highlight],
                out_dir / f"{highlight}_architecture.png",
            )
        )
    return written


def plot_gene_architecture(gene_frame: pd.DataFrame, structure: dict[str, Any], path: Path) -> Path:
    """Draw a transcript-ordered exon map with variants stacked by splice offset."""
    apply_style()
    exons = [(int(start), int(end)) for start, end in structure["exons"]]
    strand = int(structure["strand"])
    if strand == 1:
        ordered = sorted(exons)
    else:
        ordered = sorted(exons, reverse=True)
    spans, _boundaries = _exon_spans(len(ordered))
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for start, end in spans:
        ax.add_patch(
            Rectangle((start, -1.2), end - start, 2.4, facecolor="#1a1a1a", edgecolor="#1a1a1a")
        )
    ax.plot([spans[0][0], spans[-1][1]], [0, 0], color="#1a1a1a", linewidth=1, zorder=0)
    pathogenic = gene_frame["label"].eq(1)
    for mask, color, name in (
        (pathogenic, PATHOGENIC, "Pathogenic / likely pathogenic"),
        (~pathogenic, BENIGN, "Benign / likely benign"),
    ):
        part = gene_frame.loc[mask]
        if part.empty:
            continue
        xs, ys, markers = _variant_coordinates(part, spans)
        for marker in sorted(set(markers)):
            chosen = [index for index, item in enumerate(markers) if item == marker]
            ax.scatter(
                [xs[index] for index in chosen],
                [ys[index] for index in chosen],
                c=color,
                marker=marker,
                s=28,
                alpha=0.85,
                linewidths=0,
                label=name if marker == "o" else None,
                zorder=3,
            )
    ax.axhline(0, color="#1a1a1a", linewidth=0.6)
    ax.set_ylim(-48, 48)
    ax.set_yticks([-40, -20, -2, 0, 2, 20, 40])
    ax.set_ylabel("HGVS-like splice offset (clipped at ±40)")
    ax.set_xlabel("Canonical transcript, 5′ to 3′  (introns compressed)")
    ax.set_xticks([])
    title = (
        f"{structure['symbol']}  {structure.get('canonical_transcript', '')}  "
        f"({structure.get('disease', '')})"
    )
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _exon_spans(n_exons: int) -> tuple[list[tuple[float, float]], list[float]]:
    exon_width = 1.0
    intron_width = 0.45
    spans: list[tuple[float, float]] = []
    edges: list[float] = []
    cursor = 0.0
    for index in range(n_exons):
        spans.append((cursor, cursor + exon_width))
        cursor += exon_width
        if index < n_exons - 1:
            edges.append(cursor)
            cursor += intron_width
    return spans, edges


def _variant_coordinates(
    part: pd.DataFrame, spans: list[tuple[float, float]]
) -> tuple[list[float], list[float], list[str]]:
    xs: list[float] = []
    ys: list[float] = []
    markers: list[str] = []
    for row_index, (_, row) in enumerate(part.iterrows()):
        rank = int(row["nearest_exon_rank"])
        rank = min(max(rank, 0), len(spans) - 1)
        if int(row["nearest_is_donor"]) == 1:
            x_pos = spans[rank][1]
        else:
            x_pos = spans[rank][0]
        jitter = ((row_index % 7) - 3) * 0.03
        offset = float(row["hgvs_like_offset"])
        xs.append(x_pos + jitter)
        ys.append(float(np.clip(offset, -40, 40)))
        markers.append("o" if abs(offset) <= 40 else ("^" if offset > 0 else "v"))
    return xs, ys, markers


def _roc_figure(scored: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    _draw_roc(axes[0], scored, "group", "Gene held out (GroupKFold)")
    _draw_roc(axes[1], scored, "stratified", "Random split (StratifiedKFold)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _pr_figure(scored: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    _draw_pr(axes[0], scored, "group", "Gene held out (GroupKFold)")
    _draw_pr(axes[1], scored, "stratified", "Random split (StratifiedKFold)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _series(scored: pd.DataFrame, split_key: str) -> list[tuple[str, np.ndarray, str]]:
    """Curves to draw. The two rules do not depend on the split, so both panels show them."""
    specs = [
        (f"oof_lr_{split_key}_full", "Logistic regression", MODEL_COLORS["logistic_regression"]),
        (
            f"oof_hgb_{split_key}_full",
            "HistGradientBoosting",
            MODEL_COLORS["hist_gradient_boosting"],
        ),
        (f"oof_gbm_{split_key}_full", "GradientBoosting", MODEL_COLORS["gradient_boosting"]),
        ("oof_distance_rule", "Distance rule", MODEL_COLORS["distance_rule"]),
        ("oof_dinucleotide_rule", "±1/±2 dinucleotide rule", MODEL_COLORS["dinucleotide_rule"]),
    ]
    return [
        (name, scored[column].to_numpy(dtype=float), color)
        for column, name, color in specs
        if column in scored
    ]


def _draw_roc(ax: plt.Axes, scored: pd.DataFrame, split_key: str, title: str) -> None:
    y = scored["label"].to_numpy(dtype=int)
    for name, scores, color in _series(scored, split_key):
        valid = np.isfinite(scores)
        if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[valid], scores[valid])
        score = roc_auc_score(y[valid], scores[valid])
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"{name}  {score:.3f}")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _draw_pr(ax: plt.Axes, scored: pd.DataFrame, split_key: str, title: str) -> None:
    y = scored["label"].to_numpy(dtype=int)
    prevalence = float(y.mean()) if len(y) else 0
    for name, scores, color in _series(scored, split_key):
        valid = np.isfinite(scores)
        if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y[valid], scores[valid])
        score = average_precision_score(y[valid], scores[valid])
        ax.plot(recall, precision, color=color, linewidth=1.8, label=f"{name}  {score:.3f}")
    ax.axhline(
        prevalence,
        color="#bbbbbb",
        linewidth=0.8,
        linestyle="--",
        label=f"Prevalence {prevalence:.2f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _metric_bars(metrics: dict[str, Any], path: Path) -> Path:
    rows = [
        ("Distance rule", "distance_rule", "rule", "not_applicable", "fixed_rule"),
        ("±1/±2 rule", "dinucleotide_rule", "rule", "not_applicable", "fixed_rule"),
        (
            "LR, gene held out",
            "logistic_regression",
            "full",
            "group_kfold_gene",
            "train_all_eval_subset",
        ),
        (
            "HGB, gene held out",
            "hist_gradient_boosting",
            "full",
            "group_kfold_gene",
            "train_all_eval_subset",
        ),
        (
            "GBM, gene held out",
            "gradient_boosting",
            "full",
            "group_kfold_gene",
            "train_all_eval_subset",
        ),
        (
            "LR, random split",
            "logistic_regression",
            "full",
            "stratified_kfold",
            "train_all_eval_subset",
        ),
        (
            "HGB, random split",
            "hist_gradient_boosting",
            "full",
            "stratified_kfold",
            "train_all_eval_subset",
        ),
        (
            "GBM, random split",
            "gradient_boosting",
            "full",
            "stratified_kfold",
            "train_all_eval_subset",
        ),
    ]
    return _grouped_metric_bars(metrics, rows, "all", path, "All variants")


def _hard_subset_bars(metrics: dict[str, Any], path: Path) -> Path:
    rows = [
        ("Distance rule", "distance_rule", "rule", "not_applicable", "fixed_rule"),
        ("±1/±2 rule", "dinucleotide_rule", "rule", "not_applicable", "fixed_rule"),
        ("LR full, held out", "logistic_regression", "full", "group_kfold_gene", "train_on_subset"),
        (
            "HGB full, held out",
            "hist_gradient_boosting",
            "full",
            "group_kfold_gene",
            "train_on_subset",
        ),
        (
            "LR motif, held out",
            "logistic_regression",
            "motif",
            "group_kfold_gene",
            "train_on_subset",
        ),
        (
            "HGB motif, held out",
            "hist_gradient_boosting",
            "motif",
            "group_kfold_gene",
            "train_on_subset",
        ),
        (
            "HGB motif, random",
            "hist_gradient_boosting",
            "motif",
            "stratified_kfold",
            "train_on_subset",
        ),
    ]
    return _grouped_metric_bars(
        metrics,
        rows,
        "hard_3_50",
        path,
        "Hard subset only: non-canonical variants at ±3 to ±50 bp",
    )


def _grouped_metric_bars(
    metrics: dict[str, Any],
    rows: list[tuple[str, str, str, str, str]],
    subset: str,
    path: Path,
    title: str,
) -> Path:
    labels: list[str] = []
    roc: list[float] = []
    pr: list[float] = []
    brier: list[float] = []
    for label, model, feature_set, split, protocol in rows:
        found = metric_lookup(
            metrics,
            model=model,
            feature_set=feature_set,
            split=split,
            subset=subset,
            protocol=protocol,
        )
        if found is None or found.get("skipped") or "roc_auc" not in found:
            continue
        labels.append(label)
        roc.append(float(found["roc_auc"]))
        pr.append(float(found["pr_auc"]))
        brier.append(float(found["brier"]))
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    if not labels:
        ax.text(0.5, 0.5, "Not enough labeled variants in this slice", ha="center")
        ax.set_axis_off()
    else:
        index = np.arange(len(labels))
        width = 0.25
        ax.bar(index - width, roc, width, color="#1b4f72", label="ROC-AUC")
        ax.bar(index, pr, width, color="#b9770e", label="PR-AUC")
        ax.bar(index + width, brier, width, color="#7f8c8d", label="Brier (lower is better)")
        ax.set_xticks(index, labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _calibration(scored: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    y = scored["label"].to_numpy(dtype=int)
    for column, name, color in (
        (
            "oof_hgb_group_full",
            "HistGradientBoosting, gene held out",
            MODEL_COLORS["hist_gradient_boosting"],
        ),
        (
            "oof_lr_group_full",
            "Logistic regression, gene held out",
            MODEL_COLORS["logistic_regression"],
        ),
    ):
        if column not in scored:
            continue
        scores = scored[column].to_numpy(dtype=float)
        valid = np.isfinite(scores)
        if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
            continue
        bins = np.linspace(0, 1, 9)
        centers: list[float] = []
        observed: list[float] = []
        for left, right in zip(bins[:-1], bins[1:], strict=True):
            chosen = valid & (scores >= left) & (scores < right if right < 1 else scores <= right)
            if chosen.sum() < 5:
                continue
            centers.append(float(scores[chosen].mean()))
            observed.append(float(y[chosen].mean()))
        ax.plot(centers, observed, marker="o", color=color, label=name)
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed pathogenic fraction")
    ax.set_title("Calibration of gene-held-out probabilities")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _coefficients(metrics: dict[str, Any], path: Path) -> Path:
    rows = metrics.get("logistic_coefficients", [])[:12]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not rows:
        ax.set_axis_off()
    else:
        names = [row["feature"] for row in rows][::-1]
        values = [float(row["coefficient"]) for row in rows][::-1]
        colors = [PATHOGENIC if value > 0 else BENIGN for value in values]
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="#333333", linewidth=0.6)
        ax.set_xlabel("Standardized logistic coefficient")
        ax.set_title("Full-cohort logistic coefficients (descriptive, not the CV score)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _importance(metrics: dict[str, Any], path: Path) -> Path:
    rows = metrics.get("permutation_importance", [])[:12]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not rows:
        ax.set_axis_off()
    else:
        names = [row["feature"] for row in rows][::-1]
        values = [float(row["mean_auc_drop"]) for row in rows][::-1]
        ax.barh(names, values, color="#1b4f72")
        ax.set_xlabel("Mean drop in gene-held-out ROC-AUC")
        ax.set_title("Permutation importance, HistGradientBoosting")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _distance_hist(scored: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.arange(0, 82, 2)
    for label, color, name in (
        (1, PATHOGENIC, "Pathogenic / likely pathogenic"),
        (0, BENIGN, "Benign / likely benign"),
    ):
        values = scored.loc[scored["label"].eq(label), "abs_dist_to_splice"].clip(upper=80)
        ax.hist(values, bins=bins, color=color, alpha=0.55, label=name)
    ax.set_xlabel("|distance to nearest canonical junction| (values above 80 are clipped)")
    ax.set_ylabel("Variants")
    ax.set_title("ClinVar label versus distance to the splice junction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _cohort(scored: pd.DataFrame, path: Path) -> Path:
    counts = (
        scored.groupby("gene")["label"]
        .agg(pathogenic=lambda s: int((s == 1).sum()), benign=lambda s: int((s == 0).sum()))
        .sort_values("pathogenic")
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(counts.index, counts["pathogenic"], color=PATHOGENIC, label="Pathogenic / LP")
    ax.barh(
        counts.index,
        counts["benign"],
        left=counts["pathogenic"],
        color=BENIGN,
        label="Benign / LB",
    )
    ax.set_xlabel("Variants passing QC")
    ax.set_title("Cohort composition after strict QC")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _expression_heatmap(gtex: pd.DataFrame, scored: pd.DataFrame, path: Path) -> Path:
    brain = gtex[gtex["tissue"].isin(GTEX_BRAIN_TISSUES)].copy()
    if brain.empty:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        fig.savefig(path)
        plt.close(fig)
        return path
    order = [gene for gene in scored["gene"].drop_duplicates() if gene in set(brain["gene"])]
    tissues = [tissue for tissue in GTEX_BRAIN_TISSUES if tissue in set(brain["tissue"])]
    matrix = np.full((len(order), len(tissues)), np.nan)
    lookup = {(row.gene, row.tissue): row.median_tpm for row in brain.itertuples(index=False)}
    for i, gene in enumerate(order):
        for j, tissue in enumerate(tissues):
            if (gene, tissue) in lookup:
                matrix[i, j] = np.log1p(lookup[(gene, tissue)])
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    short = [tissue.replace("Brain_", "").replace("_", " ") for tissue in tissues]
    ax.set_xticks(range(len(tissues)), short, rotation=40, ha="right", fontsize=8)
    ax.set_title("GTEx v8 brain median expression, log(1 + TPM)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="log1p TPM")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def markdown_table(metrics: dict[str, Any]) -> str:
    """A Markdown table of the headline comparison, using only stored metrics."""
    header = (
        "| Model | Split | Subset | ROC-AUC | PR-AUC | Brier |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    wanted = [
        ("distance_rule", "rule", "not_applicable", "all", "fixed_rule", "Distance rule"),
        (
            "dinucleotide_rule",
            "rule",
            "not_applicable",
            "all",
            "fixed_rule",
            "±1/±2 dinucleotide rule",
        ),
        (
            "distance_logistic",
            "abs_dist_to_splice",
            "group_kfold_gene",
            "all",
            "train_all_eval_subset",
            "Distance logistic",
        ),
        (
            "logistic_regression",
            "full",
            "group_kfold_gene",
            "all",
            "train_all_eval_subset",
            "Logistic regression",
        ),
        (
            "hist_gradient_boosting",
            "full",
            "group_kfold_gene",
            "all",
            "train_all_eval_subset",
            "HistGradientBoosting",
        ),
        (
            "gradient_boosting",
            "full",
            "group_kfold_gene",
            "all",
            "train_all_eval_subset",
            "GradientBoosting",
        ),
        (
            "logistic_regression",
            "full",
            "stratified_kfold",
            "all",
            "train_all_eval_subset",
            "Logistic regression",
        ),
        (
            "hist_gradient_boosting",
            "full",
            "stratified_kfold",
            "all",
            "train_all_eval_subset",
            "HistGradientBoosting",
        ),
        (
            "gradient_boosting",
            "full",
            "stratified_kfold",
            "all",
            "train_all_eval_subset",
            "GradientBoosting",
        ),
        (
            "logistic_regression",
            "motif",
            "group_kfold_gene",
            "hard_3_50",
            "train_on_subset",
            "Logistic regression, motif features",
        ),
        (
            "hist_gradient_boosting",
            "motif",
            "group_kfold_gene",
            "hard_3_50",
            "train_on_subset",
            "HistGradientBoosting, motif features",
        ),
        (
            "hist_gradient_boosting",
            "full",
            "group_kfold_gene",
            "hard_3_50",
            "train_on_subset",
            "HistGradientBoosting, full features",
        ),
        (
            "dinucleotide_rule",
            "rule",
            "not_applicable",
            "hard_3_50",
            "fixed_rule",
            "±1/±2 rule on the hard subset",
        ),
        (
            "distance_rule",
            "rule",
            "not_applicable",
            "hard_3_50",
            "fixed_rule",
            "Distance rule on the hard subset",
        ),
    ]
    lines = [header]
    for model, feature_set, split, subset, protocol, label in wanted:
        row = metric_lookup(
            metrics,
            model=model,
            feature_set=feature_set,
            split=split,
            subset=subset,
            protocol=protocol,
        )
        if row is None:
            continue
        split_label = {
            "group_kfold_gene": "gene held out",
            "stratified_kfold": "random split",
            "not_applicable": "fixed rule",
        }.get(split, split)
        subset_label = {
            "all": "all variants",
            "hard_3_50": "±3..±50 bp",
        }.get(subset, subset)
        lines.append(
            f"| {label} | {split_label} | {subset_label} | "
            f"{format_metric(row, 'roc_auc')} | {format_metric(row, 'pr_auc')} | "
            f"{format_metric(row, 'brier')} |\n"
        )
    return "".join(lines)

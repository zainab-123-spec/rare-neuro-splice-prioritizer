"""Self-contained HTML reports for the benchmark and for one gene."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from rnsp.config import DISCLAIMER
from rnsp.model import format_metric


def render_benchmark_html(metrics: dict[str, Any], qc: dict[str, Any] | None = None) -> str:
    """Benchmark report. Every number is read from the metrics and QC documents."""
    body_rows = _metric_rows(metrics)
    qc_rows = ""
    if qc:
        drops = qc.get("drop_reasons", {})
        items = "".join(
            f"<li><code>{html.escape(str(key))}</code>: {html.escape(str(value))}</li>"
            for key, value in sorted(drops.items())
        )
        qc_rows = (
            f"<h2>Quality control</h2><p>{qc.get('n_variants', '—')} variants passed, "
            f"{qc.get('n_pathogenic', '—')} pathogenic or likely pathogenic and "
            f"{qc.get('n_benign', '—')} benign or likely benign, across "
            f"{qc.get('n_genes', '—')} genes.</p><ul>{items}</ul>"
        )
    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in metrics.get("notes", []))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Rare neuro splice prioritizer — benchmark</title>
<style>
body {{ font-family: Georgia, "Iowan Old Style", serif; max-width: 920px; margin: 2rem auto; padding: 0 1.2rem; color: #1c1c1c; line-height: 1.45; }}
h1, h2 {{ font-family: "DejaVu Sans", sans-serif; }}
table {{ border-collapse: collapse; width: 100%; font-family: "DejaVu Sans", sans-serif; font-size: 0.86rem; }}
th, td {{ border-bottom: 1px solid #ddd; text-align: left; padding: 0.35rem 0.4rem; vertical-align: top; }}
.disclaimer {{ background: #fdecea; border-left: 4px solid #b2182b; padding: 0.8rem 1rem; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Benchmark: non-coding splice variants in neurological genes</h1>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<p>Created {html.escape(str(metrics.get("created_at", "")))}.
{metrics.get("n_variants", "—")} variants,
{metrics.get("n_genes", "—")} genes,
prevalence {float(metrics.get("prevalence", 0)):.3f}.</p>
<h2>Measured out-of-fold metrics</h2>
<table>
<tr><th>Model</th><th>Split</th><th>Subset</th><th>ROC-AUC</th><th>PR-AUC</th><th>Brier</th><th>n</th></tr>
{body_rows}
</table>
<h2>How to read this</h2>
<ul>{notes}</ul>
{qc_rows}
</body>
</html>
"""


def render_gene_html(gene: str, frame: pd.DataFrame, structure: dict[str, Any]) -> str:
    """One-gene report ranked by the full-cohort deployment score."""
    part = frame[frame["gene"] == gene].copy()
    if "deployment_score" in part.columns:
        part = part.sort_values("deployment_score", ascending=False)
    rows = []
    for record in part.head(20).to_dict(orient="records"):
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('chrom')))}:{html.escape(str(record.get('pos')))}:"
            f"{html.escape(str(record.get('ref')))}&gt;{html.escape(str(record.get('alt')))}</td>"
            f"<td>{html.escape(str(record.get('clinical_significance', '')))}</td>"
            f"<td>{html.escape(str(record.get('region_class', '')))}</td>"
            f"<td>{_fmt(record.get('abs_dist_to_splice'))}</td>"
            f"<td>{_fmt(record.get('donor_maxent_delta'))}</td>"
            f"<td>{_fmt(record.get('acceptor_maxent_delta'))}</td>"
            f"<td>{_fmt(record.get('donor_pwm_delta'))}</td>"
            f"<td>{_fmt(record.get('cryptic_donor_delta'))}</td>"
            f"<td>{html.escape(str(record.get('cryptic_donor_created', '')))}</td>"
            f"<td>{_fmt(record.get('deployment_score'))}</td>"
            "</tr>"
        )
    n_pos = int(part["label"].sum()) if len(part) else 0
    n_neg = int((part["label"] == 0).sum()) if len(part) else 0
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(gene)} splice map</title>
<style>
body {{ font-family: Georgia, serif; max-width: 980px; margin: 2rem auto; padding: 0 1.2rem; color: #1c1c1c; }}
table {{ border-collapse: collapse; width: 100%; font-family: "DejaVu Sans", sans-serif; font-size: 0.82rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.3rem 0.35rem; text-align: left; }}
.disclaimer {{ background: #fdecea; border-left: 4px solid #b2182b; padding: 0.8rem 1rem; }}
</style>
</head>
<body>
<h1>{html.escape(gene)}</h1>
<p>{html.escape(str(structure.get("disease", "")))}. Canonical transcript
<code>{html.escape(str(structure.get("canonical_transcript", "")))}</code>,
strand {html.escape(str(structure.get("strand", "")))},
chr{html.escape(str(structure.get("chrom", "")))}:
{html.escape(str(structure.get("start", "")))}–{html.escape(str(structure.get("end", "")))}.</p>
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<p>{len(part)} QC-passed variants in this gene: {n_pos} pathogenic or likely pathogenic,
{n_neg} benign or likely benign. The table is sorted by the deployment score, which is a
refit on the full research cohort. It is not a cross-validated probability and it is not
a clinical classification.</p>
<table>
<tr><th>Variant</th><th>ClinVar</th><th>Region</th><th>|dist|</th>
<th>Donor MaxEnt Δ</th><th>Acceptor MaxEnt Δ</th><th>Donor PWM Δ</th>
<th>Cryptic donor Δ</th><th>Cryptic GT alert</th><th>Deployment score</th></tr>
{"".join(rows)}
</table>
</body>
</html>
"""


def _metric_rows(metrics: dict[str, Any]) -> str:
    lines = []
    for row in metrics.get("evaluations", []):
        if row.get("skipped"):
            continue
        if row.get("subset") not in {"all", "hard_3_50", "canonical_dinucleotide", "deep_gt_50"}:
            continue
        if row.get("feature_set") not in {"full", "motif", "rule", "abs_dist_to_splice"}:
            continue
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('model')))}</td>"
            f"<td>{html.escape(str(row.get('split')))}<br/>{html.escape(str(row.get('protocol')))}</td>"
            f"<td>{html.escape(str(row.get('subset')))}</td>"
            f"<td>{html.escape(format_metric(row, 'roc_auc'))}</td>"
            f"<td>{html.escape(format_metric(row, 'pr_auc'))}</td>"
            f"<td>{html.escape(format_metric(row, 'brier'))}</td>"
            f"<td>{html.escape(str(row.get('n', '')))}</td>"
            "</tr>"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.3f}"


def assert_disclaimer(document: str) -> None:
    """Used by tests: the clinical disclaimer has to be visible."""
    if DISCLAIMER not in document and html.escape(DISCLAIMER) not in document:
        raise AssertionError("report is missing the clinical disclaimer")

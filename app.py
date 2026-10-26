"""Interactive explorer for the rare-neurological splice prioritizer.

Run with ``streamlit run app.py`` after ``python -m rnsp build`` and
``python -m rnsp evaluate``. The app reads the processed tables. It does not
retrain a model and it does not provide a clinical interpretation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from rnsp.config import DISCLAIMER
from rnsp.model import format_metric, load_metrics, metric_lookup
from rnsp.plots import _exon_spans, _variant_coordinates, apply_style

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


@st.cache_data
def load_table(path: str) -> pd.DataFrame:
    """Load a variant table. ``chrom`` stays a string so X is not numeric."""
    return pd.read_csv(path, dtype={"chrom": str, "gene": str})


def main() -> None:
    """Render the explorer."""
    st.set_page_config(page_title="Rare neuro splice prioritizer", layout="wide")
    st.title("Rare neurological splice-variant prioritizer")
    st.caption(
        "Interpretable non-coding features for Mendelian neurological genes. "
        "Inspired by the splicing question in DeepMind AlphaGenome, built from "
        "ClinVar, Ensembl, GTEx, and the Human Protein Atlas."
    )
    st.warning(DISCLAIMER)

    scored_path = PROCESSED / "variants_scored.csv"
    plain_path = PROCESSED / "variants.csv"
    table_path = scored_path if scored_path.exists() else plain_path
    if not table_path.exists():
        st.error(
            "No variant table yet. From the repository root run "
            "`python -m rnsp build` and then `python -m rnsp evaluate`."
        )
        st.stop()

    frame = load_table(str(table_path))
    structures = _structures()
    metrics = _metrics()
    genes = sorted(frame["gene"].unique())
    with st.sidebar:
        st.header("Cohort")
        gene = st.selectbox(
            "Neurological gene", genes, index=genes.index("SCN1A") if "SCN1A" in genes else 0
        )
        region = st.selectbox(
            "Region class",
            ["all", "canonical_dinucleotide", "hard_3_50", "deep_gt_50", "other_near"],
        )
        query = st.text_input(
            "Genomic variant (chr:pos:ref>alt)",
            placeholder="2:166012108:C>G",
        )
        st.caption(f"Loaded {len(frame):,} QC-passed variants from {table_path.name}.")

    part = frame[frame["gene"] == gene].copy()
    if region != "all" and "region_class" in part.columns:
        part = part[part["region_class"] == region]
    matched = _match_query(part if query else frame[frame["gene"] == gene], query)

    overview, inspector, benchmark = st.tabs(["Gene map", "Variant inspector", "Benchmarks"])
    with overview:
        _gene_tab(gene, part, structures)
    with inspector:
        _inspector_tab(matched if matched is not None else part)
    with benchmark:
        _benchmark_tab(metrics, frame)


def _gene_tab(gene: str, part: pd.DataFrame, structures: dict) -> None:
    structure = structures.get(gene)
    left, right = st.columns([1.3, 1])
    with left:
        if structure is None:
            st.info("Gene structure was not saved. Re-run the build step.")
        else:
            st.subheader(f"{gene} canonical transcript")
            st.write(
                f"{structure.get('disease', '')}. "
                f"Transcript `{structure.get('canonical_transcript', '')}`, "
                f"strand {structure.get('strand')}, "
                f"chr{structure.get('chrom')}:{structure.get('start')}–{structure.get('end')}."
            )
            apply_style()
            fig = _live_architecture(part, structure)
            st.pyplot(fig, clear_figure=True)
        st.caption(
            "Exons are drawn in transcription order with compressed introns. "
            "Y is the HGVS-like offset from the nearest canonical junction. "
            "Triangles sit beyond ±40 bp."
        )
    with right:
        st.subheader("Ranked variants")
        ranked = _ranked(part)
        columns = [
            column
            for column in (
                "pos",
                "ref",
                "alt",
                "clinical_significance",
                "region_class",
                "abs_dist_to_splice",
                "donor_maxent_delta",
                "acceptor_maxent_delta",
                "cryptic_donor_created",
                "deployment_score",
            )
            if column in ranked.columns
        ]
        st.dataframe(ranked[columns].head(25), use_container_width=True, hide_index=True)
        n_pos = int(part["label"].sum()) if len(part) else 0
        st.metric("Variants in view", len(part), delta=f"{n_pos} pathogenic / LP")


def _inspector_tab(part: pd.DataFrame) -> None:
    if part.empty:
        st.info("No variant matches that filter.")
        return
    ranked = _ranked(part)
    labels = [
        f"{row.chrom}:{int(row.pos)}:{row.ref}>{row.alt}  ({row.clinical_significance})"
        for row in ranked.itertuples(index=False)
    ]
    choice = st.selectbox("Variant", range(len(labels)), format_func=lambda i: labels[i])
    row = ranked.iloc[int(choice)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("|distance|", f"{float(row['abs_dist_to_splice']):.0f} bp")
    c2.metric("Donor MaxEnt Δ", _num(row.get("donor_maxent_delta")))
    c3.metric("Acceptor MaxEnt Δ", _num(row.get("acceptor_maxent_delta")))
    c4.metric(
        "Cryptic donor alert", "yes" if int(row.get("cryptic_donor_created", 0) or 0) else "no"
    )
    st.write(
        f"**{row['gene']}** {row['chrom']}:{int(row['pos'])} {row['ref']}>{row['alt']} · "
        f"{row.get('hgvs', '')} · {row.get('region_class', '')} · "
        f"ClinVar: {row.get('clinical_significance', '')}"
    )
    if int(row.get("cryptic_donor_created", 0) or 0) or int(
        row.get("cryptic_acceptor_created", 0) or 0
    ):
        st.error(
            "Cryptic splice-site alert: the alternate allele creates a GT or AG "
            "whose MaxEntScan score is at least 3 bits and which is not an annotated junction."
        )
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [
        "Donor MaxEnt",
        "Acceptor MaxEnt",
        "Donor PWM",
        "Acceptor PWM",
        "PPT Y-content",
        "Branchpoint",
        "Cryptic donor",
        "Cryptic acceptor",
    ]
    keys = [
        "donor_maxent_delta",
        "acceptor_maxent_delta",
        "donor_pwm_delta",
        "acceptor_pwm_delta",
        "ppt_delta",
        "branchpoint_delta",
        "cryptic_donor_delta",
        "cryptic_acceptor_delta",
    ]
    values = [float(row[key]) if pd.notna(row.get(key)) else 0.0 for key in keys]
    colors = ["#b2182b" if value < 0 else "#2166ac" for value in values]
    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="#333333", linewidth=0.6)
    ax.set_xlabel("Alternate minus reference (negative weakens the annotated signal)")
    ax.set_title("Motif and tract deltas for the selected variant")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    with st.expander("Feature row"):
        st.json({key: _jsonable(row[key]) for key in row.index})


def _benchmark_tab(metrics: dict | None, frame: pd.DataFrame) -> None:
    if metrics is None:
        st.info("Metrics appear after `python -m rnsp evaluate`.")
        return
    st.subheader("Gene-held-out versus random split")
    specs = [
        ("Distance rule", "distance_rule", "rule", "not_applicable", "fixed_rule"),
        ("±1/±2 rule", "dinucleotide_rule", "rule", "not_applicable", "fixed_rule"),
        (
            "Logistic, held out",
            "logistic_regression",
            "full",
            "group_kfold_gene",
            "train_all_eval_subset",
        ),
        (
            "HGB, held out",
            "hist_gradient_boosting",
            "full",
            "group_kfold_gene",
            "train_all_eval_subset",
        ),
        (
            "Logistic, random",
            "logistic_regression",
            "full",
            "stratified_kfold",
            "train_all_eval_subset",
        ),
        (
            "HGB, random",
            "hist_gradient_boosting",
            "full",
            "stratified_kfold",
            "train_all_eval_subset",
        ),
        (
            "HGB motif, hard ±3..±50",
            "hist_gradient_boosting",
            "motif",
            "group_kfold_gene",
            "train_on_subset",
        ),
    ]
    rows = []
    for label, model, feature_set, split, protocol in specs:
        found = metric_lookup(
            metrics,
            model=model,
            feature_set=feature_set,
            split=split,
            subset="hard_3_50" if "hard" in label else "all",
            protocol=protocol,
        )
        rows.append(
            {
                "comparison": label,
                "ROC-AUC": format_metric(found, "roc_auc"),
                "PR-AUC": format_metric(found, "pr_auc"),
                "Brier": format_metric(found, "brier"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"Deployment model saved for the explorer: `{metrics.get('deployment_model', '—')}`. "
        "Its in-app score is a full-cohort refit, not the cross-validated number in this table."
    )
    st.write(
        f"Cohort: {metrics.get('n_variants')} variants, {metrics.get('n_genes')} genes, "
        f"prevalence {float(metrics.get('prevalence', 0)):.3f}."
    )
    st.write("Region classes in the loaded table:")
    if "region_class" in frame.columns:
        st.bar_chart(frame["region_class"].value_counts())


def _live_architecture(part: pd.DataFrame, structure: dict):
    exons = [(int(start), int(end)) for start, end in structure["exons"]]
    strand = int(structure["strand"])
    ordered = sorted(exons) if strand == 1 else sorted(exons, reverse=True)
    spans, _edges = _exon_spans(len(ordered))
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    for start, end in spans:
        ax.add_patch(plt.Rectangle((start, -1.2), end - start, 2.4, color="#1a1a1a"))
    if len(part):
        for label, color in ((1, "#b2182b"), (0, "#2166ac")):
            chosen = part[part["label"] == label]
            if chosen.empty:
                continue
            xs, ys, _markers = _variant_coordinates(chosen, spans)
            ax.scatter(xs, ys, c=color, s=26, linewidths=0, zorder=3)
    ax.set_ylim(-48, 48)
    ax.set_yticks([-40, -10, 0, 10, 40])
    ax.set_xticks([])
    ax.set_ylabel("Splice offset")
    ax.set_title(f"{structure['symbol']} exon map")
    fig.tight_layout()
    return fig


def _ranked(part: pd.DataFrame) -> pd.DataFrame:
    if "deployment_score" in part.columns:
        return part.sort_values("deployment_score", ascending=False)
    if "abs_dist_to_splice" in part.columns:
        return part.sort_values("abs_dist_to_splice", ascending=True)
    return part


def _match_query(part: pd.DataFrame, query: str) -> pd.DataFrame | None:
    text = query.strip().replace("chr", "")
    if not text:
        return None
    chrom = pos = ref = alt = None
    if ">" in text:
        left, alt = text.split(">", 1)
        pieces = left.split(":")
        if len(pieces) >= 3:
            chrom, pos, ref = pieces[0], pieces[1], pieces[2]
    else:
        pieces = text.split(":")
        if len(pieces) == 4:
            chrom, pos, ref, alt = pieces
    if chrom is None or pos is None or ref is None or alt is None:
        st.sidebar.error("Use chr:pos:ref>alt, for example 2:166012108:C>G.")
        return part.iloc[0:0]
    try:
        position = int(pos)
    except ValueError:
        return part.iloc[0:0]
    matched = part[
        part["chrom"].astype(str).eq(str(chrom))
        & part["pos"].eq(position)
        & part["ref"].str.upper().eq(ref.upper())
        & part["alt"].str.upper().eq(alt.upper())
    ]
    return matched


def _structures() -> dict:
    path = PROCESSED / "gene_structures.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _metrics() -> dict | None:
    path = PROCESSED / "metrics.json"
    if not path.exists():
        return None
    return load_metrics(path)


def _num(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.2f}"


def _jsonable(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()

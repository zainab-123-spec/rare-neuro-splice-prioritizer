"""Build the variant table from ClinVar, Ensembl, GTEx, and the Human Protein Atlas."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rnsp.clinvar import ClinVarVariant, collect_uids, deduplicate, fetch_summaries, parse_record
from rnsp.config import (
    CACHE_DIR,
    ENSEMBL_MIN_INTERVAL_S,
    GENE_BY_SYMBOL,
    GENE_PANEL,
    GTEX_MIN_INTERVAL_S,
    MODEL_FEATURES_FULL,
    NCBI_MIN_INTERVAL_S,
    PROCESSED_DIR,
    PanelGene,
)
from rnsp.ensembl import fetch_constrained, fetch_gene, fetch_sequence_store
from rnsp.expression import fetch_expression
from rnsp.features import compute_features
from rnsp.http import HttpClient
from rnsp.splicing import (
    ACCEPTOR_MOTIF,
    DONOR_MOTIF,
    Junction,
    build_junctions,
    extract_motif,
    fit_pwm,
    pwm_as_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class LocusBundle:
    """Fetched material for one gene, before motif scores are attached."""

    panel: PanelGene
    gene: dict[str, Any]
    variants: list[ClinVarVariant]
    junctions: list[Junction]
    expression: dict[str, Any]
    constraint: list[dict[str, float]] | None
    store: Any
    donor_motifs: list[str] = field(default_factory=list)
    acceptor_motifs: list[str] = field(default_factory=list)


def build_dataset(
    symbols: list[str] | None = None,
    *,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Download, QC, and featurize the panel. Writes the processed tables."""
    selected = _select_genes(symbols)
    cache = Path(cache_dir) if cache_dir else CACHE_DIR
    destination = Path(out_dir) if out_dir else PROCESSED_DIR
    destination.mkdir(parents=True, exist_ok=True)
    ncbi = HttpClient(cache, min_interval=NCBI_MIN_INTERVAL_S)
    ensembl = HttpClient(
        cache,
        min_interval=ENSEMBL_MIN_INTERVAL_S,
        max_retries=8,
        timeout=(10.0, 45.0),
        max_backoff=12.0,
    )
    expression_client = HttpClient(cache, min_interval=GTEX_MIN_INTERVAL_S)

    qc: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    bundles: list[LocusBundle] = []
    pending = list(selected)
    for attempt in (1, 2):
        if attempt == 2 and pending:
            logger.info("Retrying %s genes after a transient API failure", len(pending))
        still_pending: list[PanelGene] = []
        for panel in pending:
            try:
                bundle = _fetch_locus(panel, ncbi, ensembl, expression_client, qc)
            except Exception as exc:
                logger.exception("Attempt %s failed for %s", attempt, panel.symbol)
                if attempt == 2:
                    failures.append({"gene": panel.symbol, "error": str(exc)})
                    qc[f"gene_failed:{panel.symbol}"] += 1
                else:
                    still_pending.append(panel)
                continue
            if bundle.variants:
                bundles.append(bundle)
            else:
                qc[f"no_variants:{panel.symbol}"] += 1
                logger.warning("No variants passed retrieval QC for %s", panel.symbol)
        pending = still_pending

    if not bundles:
        raise RuntimeError("No genes produced variants. See the log for API or QC errors.")

    donor_pwm, acceptor_pwm, pwm_counts = _fit_panel_pwm(bundles)
    rows, tissue_rows = _score_bundles(bundles, donor_pwm, acceptor_pwm, qc)
    if not rows:
        raise RuntimeError("Every variant failed feature QC. Check reference-allele mismatches.")

    frame = pd.DataFrame(rows)
    frame["chrom"] = frame["chrom"].astype(str)
    _write_outputs(
        destination,
        frame,
        bundles,
        tissue_rows,
        donor_pwm,
        acceptor_pwm,
        pwm_counts,
        qc,
        failures,
        selected,
    )
    logger.info(
        "Wrote %s variants (%s pathogenic, %s benign) across %s genes",
        len(frame),
        int(frame["label"].sum()),
        int((frame["label"] == 0).sum()),
        frame["gene"].nunique(),
    )
    return frame


def _select_genes(symbols: list[str] | None) -> list[PanelGene]:
    if not symbols:
        return list(GENE_PANEL)
    chosen: list[PanelGene] = []
    for symbol in symbols:
        key = symbol.strip().upper()
        if key not in GENE_BY_SYMBOL:
            known = ", ".join(gene.symbol for gene in GENE_PANEL)
            raise ValueError(f"Unknown panel gene {symbol}. Choose from: {known}")
        chosen.append(GENE_BY_SYMBOL[key])
    return chosen


def _fetch_locus(
    panel: PanelGene,
    ncbi: HttpClient,
    ensembl: HttpClient,
    expression_client: HttpClient,
    qc: Counter[str],
) -> LocusBundle:
    logger.info("Building %s (%s)", panel.symbol, panel.disease)
    uids = collect_uids(ncbi, panel.symbol, panel.aliases)
    qc["clinvar_uids"] += len(uids)
    summaries = fetch_summaries(ncbi, uids) if uids else []
    parsed: list[ClinVarVariant] = []
    for record in summaries:
        variant, reason = parse_record(record, panel.symbol, panel.aliases)
        if variant is None:
            qc[reason] += 1
            continue
        parsed.append(variant)
    parsed = deduplicate(parsed, qc)
    gene = fetch_gene(ensembl, panel.symbol)
    junctions = build_junctions(gene["exons"], int(gene["strand"]))
    kept: list[ClinVarVariant] = []
    for variant in parsed:
        if variant.chrom != str(gene["chrom"]):
            qc["chromosome_mismatch"] += 1
            continue
        flank = 2_000
        if not (int(gene["start"]) - flank <= variant.pos <= int(gene["end"]) + flank):
            qc["outside_locus"] += 1
            continue
        kept.append(variant)
    store = fetch_sequence_store(
        ensembl,
        str(gene["chrom"]),
        gene["exons"],
        junctions,
        [variant.pos for variant in kept],
    )
    expression = fetch_expression(
        expression_client,
        panel.symbol,
        str(gene["ensembl_id"]),
        panel.aliases,
    )
    constraint_pad = 50
    constraint = fetch_constrained(
        ensembl,
        str(gene["chrom"]),
        int(gene["start"]) - constraint_pad,
        int(gene["end"]) + constraint_pad,
    )
    bundle = LocusBundle(
        panel=panel,
        gene=gene,
        variants=kept,
        junctions=junctions,
        expression=expression,
        constraint=constraint,
        store=store,
    )
    bundle.donor_motifs = _reference_motifs(bundle, "donor", DONOR_MOTIF)
    bundle.acceptor_motifs = _reference_motifs(bundle, "acceptor", ACCEPTOR_MOTIF)
    logger.info(
        "%s: %s variants after coordinate QC, %s donor motifs, constraint %s",
        panel.symbol,
        len(kept),
        len(bundle.donor_motifs),
        "missing" if constraint is None else f"{len(constraint)} elements",
    )
    return bundle


def _reference_motifs(bundle: LocusBundle, kind: str, offsets: tuple[int, ...]) -> list[str]:
    motifs: list[str] = []
    for junction in bundle.junctions:
        if junction.kind != kind:
            continue
        try:
            motif = extract_motif(bundle.store, junction, int(bundle.gene["strand"]), offsets)
        except KeyError:
            continue
        if motif and all(base in "ACGT" for base in motif):
            motifs.append(motif)
    return motifs


def _fit_panel_pwm(bundles: list[LocusBundle]) -> tuple[Any, Any, dict[str, int]]:
    donor_seqs = [motif for bundle in bundles for motif in bundle.donor_motifs]
    acceptor_seqs = [motif for bundle in bundles for motif in bundle.acceptor_motifs]
    donor_pwm = fit_pwm(donor_seqs, len(DONOR_MOTIF), min_sequences=15)
    acceptor_pwm = fit_pwm(acceptor_seqs, len(ACCEPTOR_MOTIF), min_sequences=15)
    return donor_pwm, acceptor_pwm, {"donor": len(donor_seqs), "acceptor": len(acceptor_seqs)}


def _score_bundles(
    bundles: list[LocusBundle],
    donor_pwm: Any,
    acceptor_pwm: Any,
    qc: Counter[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    tissue_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        by_tissue = bundle.expression.pop("by_tissue", {})
        for tissue, tpm in by_tissue.items():
            tissue_rows.append({"gene": bundle.panel.symbol, "tissue": tissue, "median_tpm": tpm})
        expression = {
            key: value
            for key, value in bundle.expression.items()
            if key not in {"gencode_id", "gtex_gene_symbol", "hpa_gene", "hpa_ensembl_id"}
            and not isinstance(value, str)
        }
        observed = bundle.constraint is not None
        logged = False
        for variant in bundle.variants:
            try:
                features = compute_features(
                    pos=variant.pos,
                    ref=variant.ref,
                    alt=variant.alt,
                    strand=int(bundle.gene["strand"]),
                    store=bundle.store,
                    junctions=bundle.junctions,
                    donor_pwm=donor_pwm,
                    acceptor_pwm=acceptor_pwm,
                    expression=expression,
                    constraint_elements=bundle.constraint,
                    constraint_observed=observed,
                )
            except ValueError as exc:
                reason = (
                    "reference_mismatch" if "reference mismatch" in str(exc) else "feature_error"
                )
                qc[reason] += 1
                logger.debug(
                    "Dropping %s %s:%s (%s)", variant.gene, variant.chrom, variant.pos, exc
                )
                continue
            except KeyError:
                qc["sequence_missing"] += 1
                continue
            if not logged:
                logger.info(
                    "Example %s %s:%s%s>%s dist=%s class=%s donor_maxent_delta=%s",
                    variant.gene,
                    variant.chrom,
                    variant.pos,
                    variant.ref,
                    variant.alt,
                    features["abs_dist_to_splice"],
                    features["region_class"],
                    features["donor_maxent_delta"],
                )
                logged = True
            rows.append(_public_row(bundle, variant, features))
            qc["passed"] += 1
    return rows, tissue_rows


def _public_row(
    bundle: LocusBundle, variant: ClinVarVariant, features: dict[str, Any]
) -> dict[str, Any]:
    review = variant.review_status.lower()
    row: dict[str, Any] = {
        "gene": variant.gene,
        "category": bundle.panel.category,
        "disease": bundle.panel.disease,
        "ensembl_id": bundle.gene["ensembl_id"],
        "canonical_transcript": bundle.gene["canonical_transcript"],
        "strand": int(bundle.gene["strand"]),
        "chrom": variant.chrom,
        "pos": variant.pos,
        "ref": variant.ref,
        "alt": variant.alt,
        "label": variant.label,
        "clinical_significance": variant.clinical_significance,
        "review_status": variant.review_status,
        "criteria_provided": int("criteria provided" in review),
        "consequences": "|".join(variant.consequences),
        "accession": variant.accession,
        "clinvar_uid": variant.uid,
        "title": variant.title,
        "hgvs": variant.hgvs,
        "hpa_rna_tissue_specificity": bundle.expression.get("hpa_rna_tissue_specificity"),
        "hpa_brain_regional_specificity": bundle.expression.get("hpa_brain_regional_specificity"),
    }
    row.update(features)
    for name in MODEL_FEATURES_FULL:
        row.setdefault(name, float("nan"))
    return row


def _write_outputs(
    destination: Path,
    frame: pd.DataFrame,
    bundles: list[LocusBundle],
    tissue_rows: list[dict[str, Any]],
    donor_pwm: Any,
    acceptor_pwm: Any,
    pwm_counts: dict[str, int],
    qc: Counter[str],
    failures: list[dict[str, str]],
    selected: list[PanelGene],
) -> None:
    frame.sort_values(["gene", "chrom", "pos"]).to_csv(destination / "variants.csv", index=False)
    if tissue_rows:
        pd.DataFrame(tissue_rows).to_csv(destination / "gtex_median_tpm.csv", index=False)
    structures = {
        bundle.panel.symbol: {
            "symbol": bundle.panel.symbol,
            "category": bundle.panel.category,
            "disease": bundle.panel.disease,
            "ensembl_id": bundle.gene["ensembl_id"],
            "canonical_transcript": bundle.gene["canonical_transcript"],
            "chrom": bundle.gene["chrom"],
            "start": bundle.gene["start"],
            "end": bundle.gene["end"],
            "strand": bundle.gene["strand"],
            "exons": bundle.gene["exons"],
            "junctions": [
                {"kind": junction.kind, "anchor": junction.anchor, "exon_rank": junction.exon_rank}
                for junction in bundle.junctions
            ],
        }
        for bundle in bundles
    }
    (destination / "gene_structures.json").write_text(
        json.dumps(structures, indent=2), encoding="utf-8"
    )
    pwm_payload = {
        "donor_offsets": list(DONOR_MOTIF),
        "acceptor_offsets": list(ACCEPTOR_MOTIF),
        "background": 0.25,
        "pseudocount": 0.5,
        "n_donor_sites": pwm_counts["donor"],
        "n_acceptor_sites": pwm_counts["acceptor"],
        "donor": pwm_as_dict(donor_pwm),
        "acceptor": pwm_as_dict(acceptor_pwm),
        "note": (
            "Independent-base information-content PWM fit on canonical junctions "
            "of the genes that passed retrieval. Labels were not used."
        ),
    }
    (destination / "pwm.json").write_text(json.dumps(pwm_payload, indent=2), encoding="utf-8")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "genes_requested": [gene.symbol for gene in selected],
        "genes_built": [bundle.panel.symbol for bundle in bundles],
        "genes_failed": failures,
        "n_variants": int(len(frame)),
        "n_pathogenic": int(frame["label"].sum()),
        "n_benign": int((frame["label"] == 0).sum()),
        "n_genes": int(frame["gene"].nunique()),
        "by_gene": _counts_by(frame, "gene"),
        "by_region_class": _counts_by(frame, "region_class"),
        "by_category": _counts_by(frame, "category"),
        "drop_reasons": dict(qc),
        "pwm": pwm_counts,
        "sources": [
            "NCBI ClinVar E-utilities esearch/esummary",
            "Ensembl REST lookup, sequence/region, overlap constrained",
            "GTEx Portal API v2 medianGeneExpression (gtex_v8)",
            "Human Protein Atlas gene JSON",
        ],
    }
    (destination / "qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _counts_by(frame: pd.DataFrame, column: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for key, part in frame.groupby(column):
        grouped[str(key)] = {
            "n": int(len(part)),
            "pathogenic": int(part["label"].sum()),
            "benign": int((part["label"] == 0).sum()),
        }
    return grouped

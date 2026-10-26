"""Assemble one interpretable feature row for a single-nucleotide variant."""

from __future__ import annotations

import math
from typing import Any

from rnsp.config import (
    CRYPTIC_FLANK_BP,
    GC_FLANK_BP,
    GTEX_CEREBELLUM,
    GTEX_CORTEX,
    GTEX_HIPPOCAMPUS,
    MODEL_FEATURES_FULL,
)
from rnsp.sequence import BaseStore, cpg_count, gc_content
from rnsp.splicing import (
    ACCEPTOR_MOTIF,
    BRANCH_STARTS,
    DONOR_MOTIF,
    PPT_MOTIF,
    PWM,
    Junction,
    apply_coding_snv,
    base_information,
    best_cryptic_acceptor,
    best_cryptic_donor,
    branchpoint_score,
    coding_window,
    cryptic_acceptor_created,
    cryptic_donor_created,
    extract_motif,
    forbidden_anchor_indexes,
    hgvs_like_offset,
    is_canonical_dinucleotide,
    nearest_junction,
    overlapping_junction,
    pwm_score,
    region_class,
    y_content,
)

NAN = float("nan")


def _delta(ref_score: float, alt_score: float) -> float:
    if math.isnan(ref_score) or math.isnan(alt_score):
        return NAN
    return alt_score - ref_score


def _motif_pair(
    store: BaseStore,
    junction: Junction,
    strand: int,
    offsets: tuple[int, ...],
    pos: int,
    ref: str,
    alt: str,
) -> tuple[str, str]:
    reference = extract_motif(store, junction, strand, offsets)
    alternate = extract_motif(store, junction, strand, offsets, pos=pos, ref=ref, alt=alt)
    return reference, alternate


def compute_features(
    *,
    pos: int,
    ref: str,
    alt: str,
    strand: int,
    store: BaseStore,
    junctions: list[Junction],
    donor_pwm: PWM,
    acceptor_pwm: PWM,
    expression: dict[str, float] | None,
    constraint_elements: list[dict[str, float]] | None,
    constraint_observed: bool,
) -> dict[str, Any]:
    """Score one SNV against canonical junctions and the local sequence.

    ``constraint_elements`` is a list of ``{start, end, score}`` dicts, or
    ``None`` when the Ensembl constrained-element request failed. Expression
    values are gene-level and may be missing.
    """
    ref = ref.upper()
    alt = alt.upper()
    genome_base = store.get(pos)
    if genome_base != ref:
        raise ValueError(f"reference mismatch at {pos}: genome {genome_base}, record {ref}")

    junction, distance, intronic = nearest_junction(pos, junctions, strand)
    canonical = is_canonical_dinucleotide(junction, distance, intronic)
    row: dict[str, Any] = {
        "abs_dist_to_splice": distance,
        "nearest_is_donor": 1 if junction.kind == "donor" else 0,
        "nearest_junction_kind": junction.kind,
        "nearest_junction_anchor": junction.anchor,
        "nearest_exon_rank": junction.exon_rank,
        "intronic_side": int(intronic),
        "hgvs_like_offset": hgvs_like_offset(junction, distance, intronic),
        "disrupts_canonical_dinucleotide": int(canonical),
        "region_class": region_class(distance, canonical),
    }

    row.update(
        _site_deltas(
            store=store,
            junctions=junctions,
            strand=strand,
            pos=pos,
            ref=ref,
            alt=alt,
            donor_pwm=donor_pwm,
            acceptor_pwm=acceptor_pwm,
        )
    )
    row.update(_cryptic_features(store, junctions, strand, pos, ref, alt))
    row.update(_composition(store, pos, ref, alt))
    row.update(_constraint(pos, constraint_elements, constraint_observed))
    row.update(_expression(expression))
    return row


def _site_deltas(
    *,
    store: BaseStore,
    junctions: list[Junction],
    strand: int,
    pos: int,
    ref: str,
    alt: str,
    donor_pwm: PWM,
    acceptor_pwm: PWM,
) -> dict[str, float]:
    out = {
        "donor_maxent_delta": 0.0,
        "acceptor_maxent_delta": 0.0,
        "donor_pwm_delta": 0.0,
        "acceptor_pwm_delta": 0.0,
        "ppt_delta": 0.0,
        "branchpoint_delta": 0.0,
        "motif_ref_information_bits": 0.0,
    }
    donor = overlapping_junction(pos, junctions, strand, "donor", DONOR_MOTIF)
    if donor is not None:
        ref_motif, alt_motif = _motif_pair(store, donor, strand, DONOR_MOTIF, pos, ref, alt)
        from rnsp.maxent import score_donor

        out["donor_maxent_delta"] = _delta(score_donor(ref_motif), score_donor(alt_motif))
        out["donor_pwm_delta"] = _delta(
            pwm_score(ref_motif, donor_pwm), pwm_score(alt_motif, donor_pwm)
        )
        out["motif_ref_information_bits"] = _reference_information(
            ref_motif, donor_pwm, DONOR_MOTIF, donor, strand, pos
        )
    acceptor = overlapping_junction(pos, junctions, strand, "acceptor", ACCEPTOR_MOTIF)
    if acceptor is not None:
        ref_motif, alt_motif = _motif_pair(store, acceptor, strand, ACCEPTOR_MOTIF, pos, ref, alt)
        from rnsp.maxent import score_acceptor

        out["acceptor_maxent_delta"] = _delta(score_acceptor(ref_motif), score_acceptor(alt_motif))
        out["acceptor_pwm_delta"] = _delta(
            pwm_score(ref_motif, acceptor_pwm), pwm_score(alt_motif, acceptor_pwm)
        )
        info = _reference_information(
            ref_motif, acceptor_pwm, ACCEPTOR_MOTIF, acceptor, strand, pos
        )
        if abs(info) > abs(out["motif_ref_information_bits"]):
            out["motif_ref_information_bits"] = info
    ppt_junction = overlapping_junction(pos, junctions, strand, "acceptor", PPT_MOTIF)
    if ppt_junction is not None:
        ref_ppt, alt_ppt = _motif_pair(store, ppt_junction, strand, PPT_MOTIF, pos, ref, alt)
        out["ppt_delta"] = _delta(y_content(ref_ppt), y_content(alt_ppt))
    branch_offsets = _branch_offsets()
    branch_junction = overlapping_junction(pos, junctions, strand, "acceptor", branch_offsets)
    if branch_junction is not None:
        ref_branch, alt_branch = _motif_pair(
            store, branch_junction, strand, branch_offsets, pos, ref, alt
        )
        out["branchpoint_delta"] = _delta(
            branchpoint_score(ref_branch), branchpoint_score(alt_branch)
        )
    return out


def _branch_offsets() -> tuple[int, ...]:
    """Every coding offset covered by a YTRAC 5-mer in the branchpoint search."""
    covered: set[int] = set()
    for start in BRANCH_STARTS:
        for step in range(5):
            covered.add(start + step)
    return tuple(sorted(covered))


def _reference_information(
    ref_motif: str,
    pwm: PWM,
    offsets: tuple[int, ...],
    junction: Junction,
    strand: int,
    pos: int,
) -> float:
    from rnsp.splicing import offset_to_genomic

    for index, offset in enumerate(offsets):
        if offset_to_genomic(junction.anchor, offset, strand, junction.kind) == pos:
            return base_information(ref_motif[index], pwm[index][ref_motif[index]])
    return 0.0


def _cryptic_features(
    store: BaseStore,
    junctions: list[Junction],
    strand: int,
    pos: int,
    ref: str,
    alt: str,
) -> dict[str, float]:
    empty = {
        "cryptic_donor_delta": NAN,
        "cryptic_acceptor_delta": NAN,
        "cryptic_donor_created": 0,
        "cryptic_acceptor_created": 0,
    }
    try:
        coding, start, end = coding_window(store, pos, CRYPTIC_FLANK_BP, strand)
        alt_coding = apply_coding_snv(coding, start, end, strand, pos, ref, alt)
    except (KeyError, ValueError):
        return empty
    donor_anchors = {junction.anchor for junction in junctions if junction.kind == "donor"}
    acceptor_anchors = {junction.anchor for junction in junctions if junction.kind == "acceptor"}
    donor_block = forbidden_anchor_indexes(donor_anchors, start, end, strand)
    acceptor_block = forbidden_anchor_indexes(acceptor_anchors, start, end, strand)
    return {
        "cryptic_donor_delta": _delta(
            best_cryptic_donor(coding, donor_block),
            best_cryptic_donor(alt_coding, donor_block),
        ),
        "cryptic_acceptor_delta": _delta(
            best_cryptic_acceptor(coding, acceptor_block),
            best_cryptic_acceptor(alt_coding, acceptor_block),
        ),
        "cryptic_donor_created": cryptic_donor_created(coding, alt_coding, donor_block),
        "cryptic_acceptor_created": cryptic_acceptor_created(coding, alt_coding, acceptor_block),
    }


def _composition(store: BaseStore, pos: int, ref: str, alt: str) -> dict[str, float]:
    start = pos - GC_FLANK_BP
    end = pos + GC_FLANK_BP
    try:
        reference = store.slice(start, end)
    except KeyError:
        return {"gc_delta": NAN, "cpg_delta": NAN, "gc_local": NAN}
    index = pos - start
    alternate = reference[:index] + alt + reference[index + 1 :]
    if reference[index] != ref:
        raise ValueError("composition window does not match the reference allele")
    gc_ref = gc_content(reference)
    gc_alt = gc_content(alternate)
    return {
        "gc_local": gc_ref,
        "gc_delta": gc_alt - gc_ref,
        "cpg_delta": float(cpg_count(alternate) - cpg_count(reference)),
    }


def _constraint(
    pos: int,
    elements: list[dict[str, float]] | None,
    observed: bool,
) -> dict[str, float]:
    if not observed or elements is None:
        return {"constraint_overlap": NAN, "constraint_score": NAN, "constraint_observed": 0.0}
    scores = [
        float(element["score"])
        for element in elements
        if int(element["start"]) <= pos <= int(element["end"])
    ]
    if not scores:
        return {"constraint_overlap": 0.0, "constraint_score": 0.0, "constraint_observed": 1.0}
    return {
        "constraint_overlap": 1.0,
        "constraint_score": max(scores),
        "constraint_observed": 1.0,
    }


def _expression(expression: dict[str, float] | None) -> dict[str, float]:
    if not expression:
        values = {
            "gtex_brain_specificity": NAN,
            "gtex_brain_mean_tpm": NAN,
            "gtex_tau": NAN,
            "gtex_cortex_tpm": NAN,
            "gtex_hippocampus_tpm": NAN,
            "gtex_cerebellum_tpm": NAN,
            "hpa_brain_ntpm": NAN,
            "expression_observed": 0.0,
        }
        return values
    return {
        "gtex_brain_specificity": expression.get("gtex_brain_specificity", NAN),
        "gtex_brain_mean_tpm": expression.get("gtex_brain_mean_tpm", NAN),
        "gtex_tau": expression.get("gtex_tau", NAN),
        "gtex_cortex_tpm": expression.get(GTEX_CORTEX, expression.get("gtex_cortex_tpm", NAN)),
        "gtex_hippocampus_tpm": expression.get(
            GTEX_HIPPOCAMPUS, expression.get("gtex_hippocampus_tpm", NAN)
        ),
        "gtex_cerebellum_tpm": expression.get(
            GTEX_CEREBELLUM, expression.get("gtex_cerebellum_tpm", NAN)
        ),
        "hpa_brain_ntpm": expression.get("hpa_brain_ntpm", NAN),
        "expression_observed": float(expression.get("expression_observed", 1.0)),
    }


def feature_defaults() -> dict[str, float]:
    """A blank numeric feature dict, useful for tests and schema checks."""
    return {name: NAN for name in MODEL_FEATURES_FULL}

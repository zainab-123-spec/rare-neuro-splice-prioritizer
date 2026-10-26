"""Exon–intron geometry and interpretable splicing scores.

Distances use a simple rule. The exon–intron boundary sits between two bases.
The adjacent base on either side has distance 1. Intronic donor +1/+2 and
acceptor -1/-2 are the canonical dinucleotide. Distance 3 through 50 is the
harder near-splice set used in the benchmark. Motif windows follow MaxEntScan:
donor -3..+6 and acceptor -20..+3, read on the coding strand.

The position-weight matrix is an independent-base information-content model
fit from the panel's own canonical junctions (pseudocount 0.5, background
0.25). It is not MaxEntScan. MaxEnt deltas are computed separately in
``rnsp.maxent`` so the two assumptions can be compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rnsp.config import (
    ACCEPTOR_OFFSETS,
    BRANCHPOINT_STARTS,
    CRYPTIC_MAXENT_ALERT,
    DONOR_OFFSETS,
    PPT_OFFSETS,
)
from rnsp.maxent import score_acceptor, score_donor
from rnsp.sequence import BaseStore, apply_snv, complement, revcomp

PWM = list[dict[str, float]]


@dataclass(frozen=True)
class Junction:
    """One canonical exon–intron junction on the coding strand."""

    kind: str
    anchor: int
    exon_rank: int


def build_junctions(exons: list[tuple[int, int]], strand: int) -> list[Junction]:
    """Build donor and acceptor anchors from genomic exon intervals.

    ``exons`` are inclusive genomic ``(start, end)`` pairs with ``start <= end``.
    Strand is +1 or -1. The anchor is the first intronic base: donor +1 or
    acceptor -1.
    """
    if strand not in (1, -1):
        raise ValueError(f"strand must be +1 or -1, got {strand}")
    if strand == 1:
        ordered = sorted(exons, key=lambda exon: (exon[0], exon[1]))
    else:
        ordered = sorted(exons, key=lambda exon: (exon[0], exon[1]), reverse=True)
    junctions: list[Junction] = []
    for index, (start, end) in enumerate(ordered):
        if index < len(ordered) - 1:
            anchor = end + 1 if strand == 1 else start - 1
            junctions.append(Junction("donor", anchor, index))
        if index > 0:
            anchor = start - 1 if strand == 1 else end + 1
            junctions.append(Junction("acceptor", anchor, index))
    return junctions


def distance_to_junction(pos: int, junction: Junction, strand: int) -> tuple[int, bool]:
    """Return ``(absolute distance, is_intronic)`` from a variant to a junction."""
    anchor = junction.anchor
    if junction.kind == "donor" and strand == 1:
        intronic = pos >= anchor
        distance = pos - anchor + 1 if intronic else anchor - pos
    elif junction.kind == "donor" and strand == -1:
        intronic = pos <= anchor
        distance = anchor - pos + 1 if intronic else pos - anchor
    elif junction.kind == "acceptor" and strand == 1:
        intronic = pos <= anchor
        distance = anchor - pos + 1 if intronic else pos - anchor
    elif junction.kind == "acceptor" and strand == -1:
        intronic = pos >= anchor
        distance = pos - anchor + 1 if intronic else anchor - pos
    else:
        raise ValueError(f"unsupported junction {junction.kind} strand {strand}")
    if distance < 1:
        raise ValueError("distance must be at least 1")
    return distance, intronic


def nearest_junction(
    pos: int, junctions: list[Junction], strand: int
) -> tuple[Junction, int, bool]:
    """Pick the closest junction. Intronic wins an exact distance tie."""
    if not junctions:
        raise ValueError("no junctions")
    best: tuple[tuple[int, int], Junction, int, bool] | None = None
    for junction in junctions:
        distance, intronic = distance_to_junction(pos, junction, strand)
        key = (distance, 0 if intronic else 1)
        if best is None or key < best[0]:
            best = (key, junction, distance, intronic)
    assert best is not None
    return best[1], best[2], best[3]


def hgvs_like_offset(junction: Junction, distance: int, intronic: bool) -> int:
    """Signed offset in the HGVS intron style, with the exonic side flipped.

    Donor intron is positive, acceptor intron is negative. The exonic side
    uses the opposite sign so the absolute value stays the distance.
    """
    if junction.kind == "donor":
        return distance if intronic else -distance
    return -distance if intronic else distance


def is_canonical_dinucleotide(junction: Junction, distance: int, intronic: bool) -> bool:
    """True at intronic donor +1/+2 or acceptor -1/-2."""
    return bool(intronic and distance in (1, 2))


def offset_to_genomic(anchor: int, offset: int, strand: int, kind: str) -> int:
    """Map a coding-strand splice offset onto a genomic coordinate."""
    if offset == 0:
        raise ValueError("splice offsets skip 0")
    if kind == "donor" and strand == 1:
        return anchor + offset - 1 if offset > 0 else anchor + offset
    if kind == "donor" and strand == -1:
        return anchor - (offset - 1) if offset > 0 else anchor - offset
    if kind == "acceptor" and strand == 1:
        return anchor + offset + 1 if offset < 0 else anchor + offset
    if kind == "acceptor" and strand == -1:
        return anchor - offset - 1 if offset < 0 else anchor - offset
    raise ValueError(f"unsupported kind {kind} strand {strand}")


def region_class(distance: int, canonical: bool) -> str:
    """Name the evaluation slice for a variant."""
    if canonical:
        return "canonical_dinucleotide"
    if 3 <= distance <= 50:
        return "hard_3_50"
    if distance > 50:
        return "deep_gt_50"
    return "other_near"


def fit_pwm(
    sequences: list[str],
    length: int,
    pseudocount: float = 0.5,
    min_sequences: int = 20,
) -> PWM:
    """Column frequencies from equal-length coding-strand motifs."""
    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    counts = [{base: pseudocount for base in "ACGT"} for _ in range(length)]
    used = 0
    for seq in sequences:
        motif = seq.upper()
        if len(motif) != length or any(base not in "ACGT" for base in motif):
            continue
        used += 1
        for index, base in enumerate(motif):
            counts[index][base] += 1
    if used < min_sequences:
        raise ValueError(f"need at least {min_sequences} motifs to fit a PWM, found {used}")
    denominator = used + 4 * pseudocount
    return [{base: counts[index][base] / denominator for base in "ACGT"} for index in range(length)]


def pwm_score(seq: str, pwm: PWM, background: float = 0.25) -> float:
    """Sum of log2(frequency / background) across a motif. Units are bits."""
    motif = seq.upper()
    if len(motif) != len(pwm):
        raise ValueError("motif length does not match the PWM")
    if any(base not in "ACGT" for base in motif):
        return float("nan")
    return sum(
        math.log2(column[base] / background) for base, column in zip(motif, pwm, strict=True)
    )


def base_information(base: str, frequency: float, background: float = 0.25) -> float:
    """Information of one reference base under the PWM column, in bits."""
    if base not in "ACGT" or frequency <= 0:
        return float("nan")
    return math.log2(frequency / background)


def extract_motif(
    store: BaseStore,
    junction: Junction,
    strand: int,
    offsets: tuple[int, ...] | list[int],
    *,
    pos: int | None = None,
    ref: str | None = None,
    alt: str | None = None,
) -> str:
    """Read a coding-strand motif. Apply an SNV when its genomic base is inside."""
    chars: list[str] = []
    for offset in offsets:
        genomic = offset_to_genomic(junction.anchor, offset, strand, junction.kind)
        base = store.get(genomic)
        if pos is not None and genomic == pos:
            if ref is None or alt is None:
                raise ValueError("ref and alt are required to apply a variant")
            if base != ref.upper():
                raise ValueError(f"reference mismatch at {genomic}: genome {base}, record {ref}")
            base = alt.upper()
        if strand == -1:
            base = complement(base)
        chars.append(base)
    return "".join(chars)


def motif_contains(pos: int, junction: Junction, strand: int, offsets: tuple[int, ...]) -> bool:
    """True when the variant sits on one base of the motif window."""
    return any(
        offset_to_genomic(junction.anchor, offset, strand, junction.kind) == pos
        for offset in offsets
    )


def overlapping_junction(
    pos: int,
    junctions: list[Junction],
    strand: int,
    kind: str,
    offsets: tuple[int, ...],
) -> Junction | None:
    """Nearest junction of ``kind`` whose motif window contains ``pos``."""
    found: list[tuple[int, Junction]] = []
    for junction in junctions:
        if junction.kind != kind:
            continue
        if motif_contains(pos, junction, strand, offsets):
            distance, _intronic = distance_to_junction(pos, junction, strand)
            found.append((distance, junction))
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[0][1]


def y_content(seq: str) -> float:
    """Fraction of pyrimidine (C or T) bases."""
    motif = seq.upper()
    if not motif or any(base not in "ACGT" for base in motif):
        return float("nan")
    return sum(base in "CT" for base in motif) / len(motif)


def branchpoint_score(seq: str) -> float:
    """Best fractional match to YTRAC in a coding-strand upstream window.

    Y is C/T and R is A/G. A perfect 5-mer scores 1.
    """
    motif = seq.upper()
    if len(motif) < 5:
        return float("nan")
    best = 0.0
    for index in range(len(motif) - 4):
        mer = motif[index : index + 5]
        if any(base not in "ACGT" for base in mer):
            continue
        matched = 0
        matched += mer[0] in "CT"
        matched += mer[1] == "T"
        matched += mer[2] in "AG"
        matched += mer[3] == "A"
        matched += mer[4] == "C"
        best = max(best, matched / 5)
    return best


def coding_window(store: BaseStore, center: int, flank: int, strand: int) -> tuple[str, int, int]:
    """Coding-strand sequence centered on a genomic position.

    The window is symmetric, so the variant stays at index ``flank`` after
    reverse complement.
    """
    start = center - flank
    end = center + flank
    genomic = store.slice(start, end)
    if strand == 1:
        return genomic, start, end
    return revcomp(genomic), start, end


def apply_coding_snv(
    coding: str,
    window_start: int,
    window_end: int,
    strand: int,
    pos: int,
    ref: str,
    alt: str,
) -> str:
    """Apply a genomic SNV to a coding-strand window."""
    if strand == 1:
        index = pos - window_start
        return apply_snv(coding, index, ref, alt)
    index = window_end - pos
    return apply_snv(coding, index, complement(ref), complement(alt))


def _safe_donor(seq: str) -> float:
    try:
        return score_donor(seq)
    except (ValueError, KeyError):
        return float("nan")


def _safe_acceptor(seq: str) -> float:
    try:
        return score_acceptor(seq)
    except (ValueError, KeyError, ZeroDivisionError):
        return float("nan")


def best_cryptic_donor(coding: str, forbidden_indexes: set[int]) -> float:
    """Maximum MaxEnt donor score outside annotated donor +1 indexes."""
    best = float("nan")
    for index in range(0, len(coding) - 8):
        plus1 = index + 3
        if plus1 in forbidden_indexes:
            continue
        value = _safe_donor(coding[index : index + 9])
        if math.isnan(value):
            continue
        if math.isnan(best) or value > best:
            best = value
    return best


def best_cryptic_acceptor(coding: str, forbidden_indexes: set[int]) -> float:
    """Maximum MaxEnt acceptor score outside annotated acceptor -1 indexes."""
    best = float("nan")
    for index in range(0, len(coding) - 22):
        minus1 = index + 19
        if minus1 in forbidden_indexes:
            continue
        value = _safe_acceptor(coding[index : index + 23])
        if math.isnan(value):
            continue
        if math.isnan(best) or value > best:
            best = value
    return best


def cryptic_donor_created(ref_coding: str, alt_coding: str, forbidden_indexes: set[int]) -> int:
    """1 when the alt allele creates a GT donor that scores at or above the alert."""
    limit = min(len(ref_coding), len(alt_coding)) - 8
    for index in range(0, max(limit, 0)):
        plus1 = index + 3
        if plus1 in forbidden_indexes:
            continue
        ref_mer = ref_coding[index : index + 9]
        alt_mer = alt_coding[index : index + 9]
        if alt_mer[3:5] == "GT" and ref_mer[3:5] != "GT":
            value = _safe_donor(alt_mer)
            if not math.isnan(value) and value >= CRYPTIC_MAXENT_ALERT:
                return 1
    return 0


def cryptic_acceptor_created(ref_coding: str, alt_coding: str, forbidden_indexes: set[int]) -> int:
    """1 when the alt allele creates an AG acceptor that clears the alert score."""
    limit = min(len(ref_coding), len(alt_coding)) - 22
    for index in range(0, max(limit, 0)):
        minus1 = index + 19
        if minus1 in forbidden_indexes:
            continue
        ref_mer = ref_coding[index : index + 23]
        alt_mer = alt_coding[index : index + 23]
        if alt_mer[18:20] == "AG" and ref_mer[18:20] != "AG":
            value = _safe_acceptor(alt_mer)
            if not math.isnan(value) and value >= CRYPTIC_MAXENT_ALERT:
                return 1
    return 0


def coding_index_to_genomic(index: int, window_start: int, window_end: int, strand: int) -> int:
    """Map an index in a coding window back to a genomic coordinate."""
    if strand == 1:
        return window_start + index
    return window_end - index


def forbidden_anchor_indexes(
    anchors: set[int], window_start: int, window_end: int, strand: int
) -> set[int]:
    """Coding indexes of annotated anchors that fall inside the window."""
    indexes: set[int] = set()
    for anchor in anchors:
        if window_start <= anchor <= window_end:
            if strand == 1:
                indexes.add(anchor - window_start)
            else:
                indexes.add(window_end - anchor)
    return indexes


def pwm_as_dict(pwm: PWM) -> list[dict[str, float]]:
    """Return a JSON-serializable copy of a PWM."""
    return [{base: float(column[base]) for base in "ACGT"} for column in pwm]


def pwm_from_dict(payload: list[dict[str, float]]) -> PWM:
    """Rebuild a PWM saved by :func:`pwm_as_dict`."""
    return [{base: float(column[base]) for base in "ACGT"} for column in payload]


# Re-exported offsets so feature code can import them from one place.
DONOR_MOTIF = DONOR_OFFSETS
ACCEPTOR_MOTIF = ACCEPTOR_OFFSETS
PPT_MOTIF = PPT_OFFSETS
BRANCH_STARTS = BRANCHPOINT_STARTS

"""Sequence geometry, PWM information, and MaxEntScan scores."""

from __future__ import annotations

import math

import pytest
from rnsp.maxent import score_acceptor, score_donor
from rnsp.sequence import (
    BaseStore,
    apply_snv,
    complement,
    covering_tiles,
    cpg_count,
    gc_content,
    merge_intervals,
    revcomp,
)
from rnsp.splicing import (
    BRANCH_STARTS,
    Junction,
    branchpoint_score,
    build_junctions,
    cryptic_donor_created,
    distance_to_junction,
    fit_pwm,
    hgvs_like_offset,
    is_canonical_dinucleotide,
    nearest_junction,
    offset_to_genomic,
    pwm_score,
    region_class,
    y_content,
)


def test_revcomp_and_complement() -> None:
    assert complement("ACGT") == "TGCA"
    assert revcomp("CAGGTAAGT") == "ACTTACCTG"
    assert revcomp(revcomp("ACGTN")) == "ACGTN"


def test_gc_and_cpg_counts() -> None:
    assert gc_content("GGCC") == 1
    assert gc_content("AATT") == 0
    assert math.isnan(gc_content(""))
    assert cpg_count("ACGCG") == 2
    assert apply_snv("AAAA", 1, "A", "G") == "AGAA"
    with pytest.raises(ValueError):
        apply_snv("AAAA", 0, "T", "G")


def test_merge_intervals_joins_touching_spans() -> None:
    assert merge_intervals([(1, 5), (6, 8), (20, 21), (3, 4)]) == [(1, 8), (20, 21)]


def test_covering_tiles_align_to_a_fixed_grid() -> None:
    assert covering_tiles([(7990, 8010), (10, 20)], 8000) == [
        (1, 8000),
        (8001, 16000),
    ]
    assert covering_tiles([(-30, 10)], 8000) == [(1, 8000)]


def test_base_store_slice_is_inclusive() -> None:
    store = BaseStore([(10, "ACGTAC")])
    assert store.get(10) == "A"
    assert store.get(15) == "C"
    assert store.slice(12, 14) == "GTA"
    assert 9 not in store


def test_maxent_donor_matches_published_examples() -> None:
    # Values are the rounded scores documented with the MaxEntScan / maxentpy models.
    assert score_donor("cagGTAAGT") == pytest.approx(10.86, abs=0.01)
    assert score_donor("gagGTAAGT") == pytest.approx(11.08, abs=0.01)
    assert score_donor("taaATAAGT") == pytest.approx(-0.12, abs=0.01)


def test_maxent_acceptor_matches_published_examples() -> None:
    assert score_acceptor("ttccaaacgaacttttgtAGgga") == pytest.approx(2.89, abs=0.01)
    assert score_acceptor("tgtctttttctgtgtggcAGtgg") == pytest.approx(8.19, abs=0.01)
    assert score_acceptor("ttctctcttcagacttatAGcaa") == pytest.approx(-0.08, abs=0.01)


def test_maxent_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        score_donor("CAGGTAAG")
    with pytest.raises(ValueError):
        score_acceptor("A" * 22)


def test_pwm_consensus_beats_scrambled_sequence() -> None:
    pwm = fit_pwm(["CAGGTAAGT"] * 30, 9)
    assert pwm_score("CAGGTAAGT", pwm) > pwm_score("TTTTTTTTT", pwm)


def test_three_exons_make_two_donors_and_two_acceptors() -> None:
    exons = [(100, 130), (200, 250), (400, 430)]
    junctions = build_junctions(exons, strand=1)
    donors = [item for item in junctions if item.kind == "donor"]
    acceptors = [item for item in junctions if item.kind == "acceptor"]
    assert [item.anchor for item in donors] == [131, 251]
    assert [item.anchor for item in acceptors] == [199, 399]


def test_plus_strand_distance_and_canonical_flag() -> None:
    junction = Junction("donor", 131, 0)
    distance, intronic = distance_to_junction(131, junction, 1)
    assert (distance, intronic) == (1, True)
    assert is_canonical_dinucleotide(junction, distance, intronic)
    distance, intronic = distance_to_junction(130, junction, 1)
    assert (distance, intronic) == (1, False)
    assert not is_canonical_dinucleotide(junction, distance, intronic)
    distance, intronic = distance_to_junction(133, junction, 1)
    assert (distance, intronic) == (3, True)
    assert region_class(3, False) == "hard_3_50"
    assert region_class(80, False) == "deep_gt_50"
    assert hgvs_like_offset(junction, 5, True) == 5
    assert offset_to_genomic(131, 1, 1, "donor") == 131
    assert offset_to_genomic(131, -3, 1, "donor") == 128


def test_minus_strand_donor_anchor() -> None:
    # Transcription reads high coordinate to low. Donor +1 is exon.start - 1.
    exons = [(101, 180), (20, 40)]
    junctions = build_junctions(exons, strand=-1)
    donor = next(item for item in junctions if item.kind == "donor")
    acceptor = next(item for item in junctions if item.kind == "acceptor")
    assert donor.anchor == 100
    assert acceptor.anchor == 41
    distance, intronic = distance_to_junction(100, donor, -1)
    assert (distance, intronic) == (1, True)
    distance, intronic = distance_to_junction(98, donor, -1)
    assert (distance, intronic) == (3, True)
    nearest, nearest_distance, _ = nearest_junction(100, junctions, -1)
    assert nearest.kind == "donor"
    assert nearest_distance == 1


def test_cryptic_gt_creation_flag() -> None:
    created = cryptic_donor_created("cagATAAGT", "cagGTAAGT", set())
    assert created == 1
    assert cryptic_donor_created("cagGTAAGT", "cagGTAAGT", set()) == 0
    assert cryptic_donor_created("cagATAAGT", "cagGTAAGT", {3}) == 0


def test_ppt_and_branchpoint_scores() -> None:
    assert y_content("CCCC") == 1
    assert y_content("AAAA") == 0
    assert branchpoint_score("CTAAC") == 1
    assert branchpoint_score("AAAAA") < 1
    assert min(BRANCH_STARTS) == -48


def test_acceptor_window_lengths() -> None:
    from rnsp.config import ACCEPTOR_OFFSETS, DONOR_OFFSETS, PPT_OFFSETS

    assert len(DONOR_OFFSETS) == 9
    assert len(ACCEPTOR_OFFSETS) == 23
    assert list(PPT_OFFSETS) == list(range(-20, -2))

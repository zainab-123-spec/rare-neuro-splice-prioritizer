"""Feature rows and ClinVar parsing, including a real esummary fixture."""

from __future__ import annotations

from collections import Counter

import pytest
from rnsp.clinvar import (
    ClinVarVariant,
    deduplicate,
    keep_consequence,
    map_label,
    parse_record,
    parse_spdi,
    search_term,
)
from rnsp.config import MODEL_FEATURES_FULL
from rnsp.features import compute_features
from rnsp.sequence import BaseStore
from rnsp.splicing import build_junctions, fit_pwm

SCN1A_RECORD = {
    "uid": "4957429",
    "accession": "VCV004957429",
    "title": "NM_001165963.4(SCN1A):c.3879+1G>C",
    "variation_set": [
        {
            "cdna_change": "NM_001165963.4(SCN1A):c.3879+1G>C",
            "variation_name": "NM_001165963.4(SCN1A):c.3879+1G>C",
            "canonical_spdi": "NC_000002.12:166012107:C:G",
            "variation_loc": [
                {
                    "status": "current",
                    "assembly_name": "GRCh38",
                    "chr": "2",
                    "start": "166012108",
                    "stop": "166012108",
                }
            ],
        }
    ],
    "germline_classification": {
        "description": "Likely pathogenic",
        "review_status": "criteria provided, single submitter",
    },
    "genes": [
        {"symbol": "LOC102724058"},
        {"symbol": "SCN1A"},
    ],
    "molecular_consequence_list": ["splice donor variant"],
}


def _store_with(edits: dict[int, str], start: int = 1, end: int = 400) -> BaseStore:
    bases = ["A"] * (end - start + 1)
    for pos, base in edits.items():
        bases[pos - start] = base
    return BaseStore([(start, "".join(bases))])


def _pwms() -> tuple[list, list]:
    donor = fit_pwm(["CAGGTAAGT"] * 25, 9)
    acceptor = fit_pwm(["T" * 18 + "AG" + "GGT"] * 25, 23)
    return donor, acceptor


def test_parse_spdi_is_one_based_and_rejects_indels() -> None:
    assert parse_spdi("NC_000002.12:166012107:C:G") == (166012108, "C", "G")
    assert parse_spdi("NC_000002.12:10:AT:A") is None
    assert parse_spdi("NC_000002.12:10:A:") is None


def test_map_label_excludes_conflicting_and_vus() -> None:
    assert map_label("Pathogenic") == 1
    assert map_label("Likely benign") == 0
    assert map_label("Benign/Likely benign") == 0
    assert map_label("Conflicting classifications of pathogenicity") is None
    assert map_label("Uncertain significance") is None
    assert map_label("") is None


def test_parse_real_scn1a_splice_donor_record() -> None:
    variant, reason = parse_record(SCN1A_RECORD, "SCN1A")
    assert reason == ""
    assert variant is not None
    assert variant.chrom == "2"
    assert variant.pos == 166012108
    assert (variant.ref, variant.alt) == ("C", "G")
    assert variant.label == 1
    assert variant.gene == "SCN1A"
    assert "splice donor variant" in variant.consequences


def test_keep_consequence_drops_coding_labeled_as_utr_on_another_transcript() -> None:
    # SCN1A c.1615G>T p.Glu539Ter is nonsense on the MANE transcript and a
    # 5' UTR variant on another transcript (ClinVar uid 5018844).
    mixed = ("nonsense", "5 prime UTR variant", "non-coding transcript variant")
    assert keep_consequence(mixed) is False
    assert keep_consequence(("5 prime UTR variant",)) is True
    assert keep_consequence(("3 prime UTR variant", "non-coding transcript variant")) is True
    assert keep_consequence(("intron variant", "5 prime UTR variant")) is True
    assert keep_consequence(("inframe_indel", "5 prime UTR variant")) is False
    record = {
        **SCN1A_RECORD,
        "uid": "5018844",
        "title": "NM_001165963.4(SCN1A):c.1615G>T (p.Glu539Ter)",
        "molecular_consequence_list": list(mixed),
    }
    variant, reason = parse_record(record, "SCN1A")
    assert variant is None
    assert reason == "consequence_excluded"


def test_utr_search_excludes_coding_phrases_and_intron_search_does_not() -> None:
    utr = search_term("SCN1A", True, "5 prime UTR variant")
    intron = search_term("SCN1A", False, "intron variant")
    assert 'NOT "missense variant"[Molecular Consequence]' in utr
    assert 'NOT "frameshift variant"[Molecular Consequence]' in utr
    assert "NOT " not in intron
    assert '"intron variant"[Molecular Consequence]' in intron


def test_parse_record_drops_wrong_gene_and_missense_only() -> None:
    wrong = dict(SCN1A_RECORD)
    wrong["genes"] = [{"symbol": "TTN"}]
    variant, reason = parse_record(wrong, "SCN1A")
    assert variant is None
    assert reason == "gene_not_on_record"
    missense = {
        **SCN1A_RECORD,
        "molecular_consequence_list": ["missense variant"],
    }
    variant, reason = parse_record(missense, "SCN1A")
    assert variant is None
    assert reason == "consequence_excluded"


def test_dedup_drops_conflicting_labels_and_keeps_one_copy() -> None:
    def make(label: int, pos: int = 10) -> ClinVarVariant:
        return ClinVarVariant(
            uid=str(pos + label),
            accession="VCV1",
            gene="SCN1A",
            chrom="2",
            pos=pos,
            ref="A",
            alt="G",
            label=label,
            clinical_significance="x",
            review_status="criteria provided",
            consequences=("intron variant",),
            title="",
            hgvs="",
        )

    qc: Counter[str] = Counter()
    kept = deduplicate([make(1), make(0), make(1, pos=20), make(1, pos=20)], qc)
    assert qc["conflicting_labels"] == 2
    assert len(kept) == 1
    assert kept[0].pos == 20
    assert qc["duplicate_records"] == 1


def test_plus_strand_donor_disruption_is_negative() -> None:
    # Exons 100-130 and 200-250. Donor +1 is 131. Consensus cagGTAAGT sits on it.
    edits = {
        128: "C",
        129: "A",
        130: "G",
        131: "G",
        132: "T",
        133: "A",
        134: "A",
        135: "G",
        136: "T",
    }
    store = _store_with(edits, start=50, end=320)
    junctions = build_junctions([(100, 130), (200, 250)], strand=1)
    donor_pwm, acceptor_pwm = _pwms()
    row = compute_features(
        pos=131,
        ref="G",
        alt="A",
        strand=1,
        store=store,
        junctions=junctions,
        donor_pwm=donor_pwm,
        acceptor_pwm=acceptor_pwm,
        expression=None,
        constraint_elements=[{"start": 120, "end": 140, "score": 12.5}],
        constraint_observed=True,
    )
    assert row["abs_dist_to_splice"] == 1
    assert row["disrupts_canonical_dinucleotide"] == 1
    assert row["region_class"] == "canonical_dinucleotide"
    assert row["donor_maxent_delta"] < -5
    assert row["donor_pwm_delta"] < 0
    assert row["constraint_overlap"] == 1
    assert row["constraint_score"] == 12.5
    assert row["expression_observed"] == 0


def test_minus_strand_genomic_allele_is_complemented() -> None:
    # Coding CAGGTAAGT at a minus-strand donor whose +1 genomic base is C.
    edits = {95: "A", 96: "C", 97: "T", 98: "T", 99: "A", 100: "C", 101: "C", 102: "T", 103: "G"}
    store = _store_with(edits, start=1, end=250)
    junctions = build_junctions([(101, 180), (20, 40)], strand=-1)
    donor_pwm, acceptor_pwm = _pwms()
    row = compute_features(
        pos=100,
        ref="C",
        alt="T",
        strand=-1,
        store=store,
        junctions=junctions,
        donor_pwm=donor_pwm,
        acceptor_pwm=acceptor_pwm,
        expression={
            "gtex_brain_specificity": 3.0,
            "expression_observed": 1.0,
            "hpa_brain_ntpm": 7.9,
        },
        constraint_elements=[],
        constraint_observed=True,
    )
    assert row["abs_dist_to_splice"] == 1
    assert row["disrupts_canonical_dinucleotide"] == 1
    assert row["donor_maxent_delta"] < -5
    assert row["gtex_brain_specificity"] == 3.0
    assert row["hpa_brain_ntpm"] == 7.9
    assert row["constraint_overlap"] == 0.0


def test_reference_mismatch_is_rejected() -> None:
    store = _store_with({131: "G"}, start=50, end=320)
    junctions = build_junctions([(100, 130), (200, 250)], strand=1)
    donor_pwm, acceptor_pwm = _pwms()
    with pytest.raises(ValueError, match="reference mismatch"):
        compute_features(
            pos=131,
            ref="A",
            alt="T",
            strand=1,
            store=store,
            junctions=junctions,
            donor_pwm=donor_pwm,
            acceptor_pwm=acceptor_pwm,
            expression=None,
            constraint_elements=None,
            constraint_observed=False,
        )


def test_ppt_delta_when_a_pyrimidine_is_lost() -> None:
    # Plus-strand acceptor anchor is 199. PPT offsets -20..-3 cover 180..196.
    edits = {pos: "T" for pos in range(180, 200)}
    edits[199] = "G"
    edits[198] = "A"
    store = _store_with(edits, start=50, end=320)
    junctions = build_junctions([(100, 130), (200, 250)], strand=1)
    donor_pwm, acceptor_pwm = _pwms()
    row = compute_features(
        pos=190,
        ref="T",
        alt="A",
        strand=1,
        store=store,
        junctions=junctions,
        donor_pwm=donor_pwm,
        acceptor_pwm=acceptor_pwm,
        expression=None,
        constraint_elements=None,
        constraint_observed=False,
    )
    assert row["region_class"] == "hard_3_50"
    assert row["disrupts_canonical_dinucleotide"] == 0
    assert row["ppt_delta"] < 0


def test_cryptic_donor_alert_on_a_created_gt() -> None:
    # Put the variant inside a run of sequence that becomes cagGTAAGT.
    # Genomic positions 100-108 are the 9-mer, variant at the G of GT (index 103).
    edits = {
        100: "C",
        101: "A",
        102: "G",
        103: "A",
        104: "T",
        105: "A",
        106: "A",
        107: "G",
        108: "T",
    }
    store = _store_with(edits, start=1, end=300)
    # Junctions far from this 9-mer so the new GT is cryptic, not the annotated donor.
    junctions = build_junctions([(10, 20), (250, 270)], strand=1)
    donor_pwm, acceptor_pwm = _pwms()
    row = compute_features(
        pos=103,
        ref="A",
        alt="G",
        strand=1,
        store=store,
        junctions=junctions,
        donor_pwm=donor_pwm,
        acceptor_pwm=acceptor_pwm,
        expression=None,
        constraint_elements=None,
        constraint_observed=False,
    )
    assert row["cryptic_donor_created"] == 1
    assert row["cryptic_donor_delta"] > 0


def test_required_model_features_cover_the_biology() -> None:
    required = {
        "abs_dist_to_splice",
        "disrupts_canonical_dinucleotide",
        "donor_maxent_delta",
        "acceptor_maxent_delta",
        "donor_pwm_delta",
        "acceptor_pwm_delta",
        "cryptic_donor_delta",
        "cryptic_acceptor_delta",
        "ppt_delta",
        "branchpoint_delta",
        "gc_delta",
        "cpg_delta",
        "motif_ref_information_bits",
        "constraint_overlap",
        "gtex_brain_specificity",
        "hpa_brain_ntpm",
    }
    assert required <= set(MODEL_FEATURES_FULL)


def test_unobserved_constraint_is_missing_not_zero() -> None:
    store = _store_with({131: "G"}, start=50, end=320)
    junctions = build_junctions([(100, 130), (200, 250)], strand=1)
    donor_pwm, acceptor_pwm = _pwms()
    row = compute_features(
        pos=131,
        ref="G",
        alt="A",
        strand=1,
        store=store,
        junctions=junctions,
        donor_pwm=donor_pwm,
        acceptor_pwm=acceptor_pwm,
        expression=None,
        constraint_elements=None,
        constraint_observed=False,
    )
    assert row["constraint_observed"] == 0
    assert row["constraint_overlap"] != row["constraint_overlap"]

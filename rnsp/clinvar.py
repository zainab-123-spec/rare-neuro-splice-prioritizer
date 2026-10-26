"""NCBI ClinVar E-utilities client and record parser.

Searches are limited to non-coding splice, intron, and UTR molecular
consequences with a pathogenic/likely pathogenic or benign/likely benign
germline classification. Coordinates are taken from the GRCh38 variation
location and checked against the canonical SPDI allele, which is 0-based.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rnsp.config import (
    CONSEQUENCE_RETMAX,
    CONSEQUENCES,
    CONTACT_EMAIL,
    NCBI_EUTILS,
    NEGATIVE_LABELS,
    POSITIVE_LABELS,
    TOOL_NAME,
)
from rnsp.http import HttpClient

logger = logging.getLogger(__name__)

ALLOWED_CONSEQUENCES = frozenset(CONSEQUENCES)

# Coding effects on some transcript. A nonsense variant can also be labeled
# "5 prime UTR variant" on a different transcript; those are not the non-coding
# alleles this project studies. Splice and intron consequences are kept even
# when a coding label is also present, because an exon-edge base can be both.
_CODING_TOKENS = (
    "missense",
    "nonsense",
    "frameshift",
    "synonymous",
    "stop gained",
    "stop lost",
    "inframe",
    "initiator codon",
    "start lost",
    "protein altering",
)
_SPLICE_OR_INTRON_TOKENS = ("splice donor", "splice acceptor", "splice region", "intron")

# Dropped from UTR esearch so a nonsense allele that is a 5' UTR on a second
# transcript does not fill the retmax window. Phrases match ClinVar's index.
UTR_CODING_EXCLUSIONS: tuple[str, ...] = (
    "missense variant",
    "nonsense",
    "frameshift variant",
    "synonymous variant",
    "inframe deletion",
    "inframe insertion",
    "inframe_indel",
    "stop lost",
    "initiator codon variant",
)


@dataclass(frozen=True)
class ClinVarVariant:
    """One QC-eligible SNV parsed from a ClinVar esummary record."""

    uid: str
    accession: str
    gene: str
    chrom: str
    pos: int
    ref: str
    alt: str
    label: int
    clinical_significance: str
    review_status: str
    consequences: tuple[str, ...]
    title: str
    hgvs: str


def keep_consequence(consequences: tuple[str, ...]) -> bool:
    """True for splice/intron alleles, and for UTR alleles that are not coding elsewhere."""
    lowered = tuple(item.lower() for item in consequences)
    if any(any(token in item for token in _SPLICE_OR_INTRON_TOKENS) for item in lowered):
        return True
    has_utr = any("utr" in item for item in lowered)
    has_coding = any(any(token in item for token in _CODING_TOKENS) for item in lowered)
    return has_utr and not has_coding


def map_label(description: str) -> int | None:
    """Map a ClinVar germline description to 1, 0, or None when it is excluded."""
    text = " ".join(description.strip().lower().split())
    if not text or "conflict" in text or "uncertain" in text:
        return None
    if text in POSITIVE_LABELS:
        return 1
    if text in NEGATIVE_LABELS:
        return 0
    return None


def parse_spdi(spdi: str) -> tuple[int, str, str] | None:
    """Parse a substitution SPDI into a 1-based position, ref, and alt.

    SPDI positions are 0-based. Insertions, deletions, and alleles that are
    not a single A/C/G/T base return None so the cohort stays SNVs.
    """
    parts = spdi.split(":")
    if len(parts) != 4:
        return None
    _accession, pos_text, deleted, inserted = parts
    if not pos_text.lstrip("-").isdigit():
        return None
    deleted = deleted.upper()
    inserted = inserted.upper()
    if len(deleted) != 1 or len(inserted) != 1:
        return None
    if deleted not in "ACGT" or inserted not in "ACGT" or deleted == inserted:
        return None
    return int(pos_text) + 1, deleted, inserted


def search_term(symbol: str, pathogenic: bool, consequence: str) -> str:
    """Build one ClinVar esearch term.

    UTR queries exclude coding consequence phrases. ClinVar often tags a
    coding allele as a 5' UTR variant on a different transcript, and those
    records otherwise occupy the whole retmax window.
    """
    if pathogenic:
        significance = (
            '(pathogenic[Clinical Significance] OR "likely pathogenic"[Clinical Significance])'
        )
    else:
        significance = '(benign[Clinical Significance] OR "likely benign"[Clinical Significance])'
    term = f'{symbol}[gene] AND {significance} AND "{consequence}"[Molecular Consequence]'
    if "utr" in consequence.lower():
        excluded = " ".join(
            f'NOT "{phrase}"[Molecular Consequence]' for phrase in UTR_CODING_EXCLUSIONS
        )
        term = f"{term} {excluded}"
    return term


def collect_uids(client: HttpClient, symbol: str, aliases: tuple[str, ...] = ()) -> list[str]:
    """Search every panel symbol and alias, consequence, and label. Return unique UIDs."""
    symbols = (symbol, *aliases)
    found: list[str] = []
    seen: set[str] = set()
    for query_symbol in symbols:
        for pathogenic in (True, False):
            for consequence in CONSEQUENCES:
                term = search_term(query_symbol, pathogenic, consequence)
                retmax = CONSEQUENCE_RETMAX[consequence]
                payload = client.get_json(
                    f"{NCBI_EUTILS}/esearch.fcgi",
                    params={
                        "db": "clinvar",
                        "term": term,
                        "retmode": "json",
                        "retmax": retmax,
                        "tool": TOOL_NAME,
                        "email": CONTACT_EMAIL,
                    },
                )
                result = payload.get("esearchresult", {})
                id_list = result.get("idlist", [])
                count = int(result.get("count", 0))
                logger.info(
                    "ClinVar %s %s %s: %s hits, kept %s",
                    query_symbol,
                    "P/LP" if pathogenic else "B/LB",
                    consequence,
                    count,
                    len(id_list),
                )
                for uid in id_list:
                    if uid not in seen:
                        seen.add(uid)
                        found.append(uid)
    return found


def fetch_summaries(
    client: HttpClient, uids: list[str], batch_size: int = 60
) -> list[dict[str, Any]]:
    """Fetch esummary documents in batches. Order follows ``uids``."""
    records: list[dict[str, Any]] = []
    for start in range(0, len(uids), batch_size):
        chunk = uids[start : start + batch_size]
        payload = client.get_json(
            f"{NCBI_EUTILS}/esummary.fcgi",
            params={
                "db": "clinvar",
                "id": ",".join(chunk),
                "retmode": "json",
                "tool": TOOL_NAME,
                "email": CONTACT_EMAIL,
            },
        )
        result = payload.get("result", {})
        for uid in result.get("uids", chunk):
            record = result.get(str(uid))
            if isinstance(record, dict):
                records.append(record)
    return records


def parse_record(
    record: dict[str, Any],
    symbol: str,
    aliases: tuple[str, ...] = (),
) -> tuple[ClinVarVariant | None, str]:
    """Parse one esummary object. The second value is a drop reason when parsing fails."""
    accepted = {symbol.upper(), *(alias.upper() for alias in aliases)}
    genes = record.get("genes") or []
    symbols = {str(gene.get("symbol", "")).upper() for gene in genes}
    if symbols.isdisjoint(accepted):
        return None, "gene_not_on_record"

    label = map_label(str((record.get("germline_classification") or {}).get("description", "")))
    if label is None:
        return None, "label_excluded"

    consequences = tuple(
        str(item).strip().lower()
        for item in (record.get("molecular_consequence_list") or [])
        if str(item).strip()
    )
    if not keep_consequence(consequences):
        return None, "consequence_excluded"

    located = _grch38_snv(record)
    if located is None:
        return None, "not_snv_or_missing_grch38"
    chrom, pos, ref, alt = located

    variation = _first_variation(record)
    hgvs = ""
    if variation is not None:
        hgvs = str(variation.get("cdna_change") or variation.get("variation_name") or "")
    review = str((record.get("germline_classification") or {}).get("review_status", ""))
    variant = ClinVarVariant(
        uid=str(record.get("uid", "")),
        accession=str(record.get("accession", "")),
        gene=symbol,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        label=label,
        clinical_significance=str(
            (record.get("germline_classification") or {}).get("description", "")
        ),
        review_status=review,
        consequences=consequences,
        title=str(record.get("title", "")),
        hgvs=hgvs,
    )
    return variant, ""


def deduplicate(variants: list[ClinVarVariant], qc: Counter[str]) -> list[ClinVarVariant]:
    """Drop duplicate alleles. If the same allele has both labels, drop all copies."""
    grouped: dict[tuple[str, str, int, str, str], list[ClinVarVariant]] = {}
    for variant in variants:
        key = (variant.gene, variant.chrom, variant.pos, variant.ref, variant.alt)
        grouped.setdefault(key, []).append(variant)
    kept: list[ClinVarVariant] = []
    for copies in grouped.values():
        labels = {item.label for item in copies}
        if len(labels) > 1:
            qc["conflicting_labels"] += len(copies)
            continue
        qc["duplicate_records"] += len(copies) - 1
        kept.append(copies[0])
    return kept


def _first_variation(record: dict[str, Any]) -> dict[str, Any] | None:
    variation_set = record.get("variation_set") or []
    if not variation_set:
        return None
    first = variation_set[0]
    return first if isinstance(first, dict) else None


def _grch38_snv(record: dict[str, Any]) -> tuple[str, int, str, str] | None:
    variation = _first_variation(record)
    if variation is None:
        return None
    parsed = parse_spdi(str(variation.get("canonical_spdi") or ""))
    if parsed is None:
        return None
    spdi_pos, ref, alt = parsed
    for loc in variation.get("variation_loc") or []:
        if str(loc.get("assembly_name")) != "GRCh38":
            continue
        if str(loc.get("status", "current")) not in {"current", ""}:
            continue
        chrom = str(loc.get("chr", "")).replace("chr", "")
        if chrom in {"", "Un", "NT"} or "_" in chrom:
            continue
        try:
            start = int(loc.get("start"))
            stop = int(loc.get("stop"))
        except (TypeError, ValueError):
            return None
        if start != stop or start != spdi_pos:
            return None
        return chrom, start, ref, alt
    return None

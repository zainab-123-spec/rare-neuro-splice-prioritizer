"""Brain-expression features from GTEx Portal API v2 and the Human Protein Atlas.

GTEx supplies median TPM by tissue (dataset ``gtex_v8``). Specificity is the
ratio of mean brain TPM to mean TPM across every returned tissue. Tau is the
standard tissue-specificity index of Yanai et al. The Human Protein Atlas gene
JSON supplies the brain RNA value in its tissue nTPM map. Both are gene-level
features: they do not separate two variants inside the same gene.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from rnsp.config import GTEX_API, GTEX_BRAIN_TISSUES, HPA_GENE_URL
from rnsp.http import HttpClient, HttpError

logger = logging.getLogger(__name__)

NAN = float("nan")


def tau_specificity(values: list[float]) -> float:
    """Yanai tau. 0 is broad expression, 1 is specific to one tissue."""
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return NAN
    peak = max(clean)
    if peak <= 0:
        return 0.0
    return sum(1.0 - (value / peak) for value in clean) / (len(clean) - 1)


def brain_expression_features(records: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize GTEx median-expression rows for one gene."""
    by_tissue: dict[str, float] = {}
    for record in records:
        tissue = str(record.get("tissueSiteDetailId", ""))
        try:
            median = float(record.get("median"))
        except (TypeError, ValueError):
            continue
        by_tissue[tissue] = median
    if not by_tissue:
        raise ValueError("GTEx returned no median expression rows")
    all_values = list(by_tissue.values())
    brain_values = [by_tissue[name] for name in GTEX_BRAIN_TISSUES if name in by_tissue]
    if not brain_values:
        raise ValueError("GTEx response did not include the expected brain tissues")
    brain_mean = sum(brain_values) / len(brain_values)
    all_mean = sum(all_values) / len(all_values)
    return {
        "gtex_brain_mean_tpm": brain_mean,
        "gtex_all_mean_tpm": all_mean,
        "gtex_brain_specificity": (brain_mean + 1e-6) / (all_mean + 1e-6),
        "gtex_tau": tau_specificity(all_values),
        "gtex_cortex_tpm": by_tissue.get("Brain_Cortex", NAN),
        "gtex_hippocampus_tpm": by_tissue.get("Brain_Hippocampus", NAN),
        "gtex_cerebellum_tpm": by_tissue.get("Brain_Cerebellum", NAN),
        "gtex_top_brain_tpm": max(brain_values),
        "gtex_n_tissues": float(len(all_values)),
    }


def expression_query_symbols(symbol: str, aliases: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Symbol first, then aliases, without duplicates. GTEx still calls GBA1 ``GBA``."""
    ordered: list[str] = []
    for name in (symbol, *aliases):
        cleaned = name.strip()
        if cleaned and cleaned.upper() not in {item.upper() for item in ordered}:
            ordered.append(cleaned)
    return tuple(ordered)


def fetch_gtex(client: HttpClient, symbol: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolve a symbol to a versioned GENCODE id and download median TPM."""
    last_error: LookupError | None = None
    for query in expression_query_symbols(symbol, aliases):
        try:
            return _fetch_gtex_symbol(client, query)
        except LookupError as exc:
            last_error = exc
            logger.info("GTEx has no exact match for %s; trying the next symbol", query)
    raise LookupError(f"GTEx gene search found no exact match for {symbol}") from last_error


def _fetch_gtex_symbol(client: HttpClient, symbol: str) -> dict[str, Any]:
    """Download median TPM for one exact GTEx gene symbol."""
    search = client.get_json(
        f"{GTEX_API}/reference/geneSearch",
        params={
            "geneId": symbol,
            "gencodeVersion": "v26",
            "genomeBuild": "GRCh38/hg38",
            "itemsPerPage": 10,
        },
    )
    rows = search.get("data") or []
    match = None
    for row in rows:
        if str(row.get("geneSymbol", "")).upper() == symbol.upper():
            match = row
            break
    if match is None:
        raise LookupError(f"GTEx gene search found no exact match for {symbol}")
    gencode_id = str(match["gencodeId"])
    expression = client.get_json(
        f"{GTEX_API}/expression/medianGeneExpression",
        params={"gencodeId": gencode_id, "datasetId": "gtex_v8", "itemsPerPage": 250},
    )
    rows = expression.get("data") or []
    features = brain_expression_features(rows)
    features["gencode_id"] = gencode_id
    features["gtex_gene_symbol"] = str(match.get("geneSymbol", symbol))
    by_tissue: dict[str, float] = {}
    for row in rows:
        tissue = str(row.get("tissueSiteDetailId", ""))
        try:
            by_tissue[tissue] = float(row.get("median"))
        except (TypeError, ValueError):
            continue
    features["by_tissue"] = by_tissue
    return features


def parse_hpa(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull brain nTPM and specificity labels from a Human Protein Atlas gene document."""
    ntpm = payload.get("RNA tissue specific nTPM") or {}
    brain_raw = ntpm.get("brain") if isinstance(ntpm, dict) else None
    try:
        brain = float(brain_raw) if brain_raw not in (None, "") else NAN
    except (TypeError, ValueError):
        brain = NAN
    return {
        "hpa_gene": payload.get("Gene"),
        "hpa_ensembl_id": payload.get("Ensembl"),
        "hpa_rna_tissue_specificity": payload.get("RNA tissue specificity"),
        "hpa_brain_regional_specificity": payload.get("RNA brain regional specificity"),
        "hpa_brain_ntpm": brain,
    }


def fetch_hpa(client: HttpClient, ensembl_id: str) -> dict[str, Any]:
    """Download the Human Protein Atlas JSON record for an Ensembl gene id."""
    gene_id = ensembl_id.split(".")[0]
    payload = client.get_json(HPA_GENE_URL.format(ensembl_id=gene_id))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected HPA payload for {gene_id}")
    return parse_hpa(payload)


def fetch_expression(
    client: HttpClient,
    symbol: str,
    ensembl_id: str,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Combine GTEx and HPA. A failure of one source does not erase the other."""
    combined: dict[str, Any] = {"expression_observed": 0.0}
    try:
        combined.update(fetch_gtex(client, symbol, aliases))
        combined["expression_observed"] = 1.0
    except (HttpError, LookupError, ValueError) as exc:
        logger.warning("GTEx failed for %s: %s", symbol, exc)
    try:
        hpa = fetch_hpa(client, ensembl_id)
        combined.update(hpa)
        if not math.isnan(float(hpa.get("hpa_brain_ntpm", NAN))):
            combined["expression_observed"] = 1.0
    except (HttpError, ValueError) as exc:
        logger.warning("Human Protein Atlas failed for %s: %s", symbol, exc)
    return combined

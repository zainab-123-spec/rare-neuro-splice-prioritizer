"""Ensembl REST lookups: canonical exons, reference sequence, constrained elements."""

from __future__ import annotations

import logging
from typing import Any

from rnsp.config import ENSEMBL_REST, ENSEMBL_TILE_BP, JUNCTION_FLANK_BP, SEQUENCE_FLANK_BP
from rnsp.http import HttpClient, HttpError
from rnsp.sequence import BaseStore, covering_tiles
from rnsp.splicing import Junction

logger = logging.getLogger(__name__)


def fetch_gene(client: HttpClient, symbol: str) -> dict[str, Any]:
    """Return locus coordinates, strand, and canonical-transcript exons."""
    meta = client.get_json(
        f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{symbol}",
        headers={"Accept": "application/json"},
    )
    transcript_id = str(meta["canonical_transcript"]).split(".")[0]
    transcript = client.get_json(
        f"{ENSEMBL_REST}/lookup/id/{transcript_id}",
        params={"expand": 1},
        headers={"Accept": "application/json"},
    )
    exons = [(int(exon["start"]), int(exon["end"])) for exon in transcript.get("Exon", [])]
    if len(exons) < 2:
        raise ValueError(f"{symbol} canonical transcript {transcript_id} has fewer than 2 exons")
    return {
        "symbol": symbol,
        "ensembl_id": str(meta["id"]),
        "chrom": str(meta["seq_region_name"]),
        "start": int(meta["start"]),
        "end": int(meta["end"]),
        "strand": int(meta["strand"]),
        "canonical_transcript": str(transcript["id"]),
        "exons": exons,
    }


def fetch_sequence_store(
    client: HttpClient,
    chrom: str,
    exons: list[tuple[int, int]],
    junctions: list[Junction],
    positions: list[int],
) -> BaseStore:
    """Fetch only the tiles needed for junctions and variant windows."""
    intervals: list[tuple[int, int]] = []
    for start, end in exons:
        intervals.append((start - JUNCTION_FLANK_BP, start + JUNCTION_FLANK_BP))
        intervals.append((end - JUNCTION_FLANK_BP, end + JUNCTION_FLANK_BP))
    for junction in junctions:
        intervals.append((junction.anchor - JUNCTION_FLANK_BP, junction.anchor + JUNCTION_FLANK_BP))
    for pos in positions:
        intervals.append((pos - SEQUENCE_FLANK_BP, pos + SEQUENCE_FLANK_BP))
    blocks: list[tuple[int, str]] = []
    for start, end in covering_tiles(intervals, ENSEMBL_TILE_BP):
        sequence = _fetch_region(client, chrom, start, end)
        blocks.append((start, sequence))
    return BaseStore(blocks)


def fetch_constrained(
    client: HttpClient, chrom: str, start: int, end: int
) -> list[dict[str, float]] | None:
    """Return GERP constrained elements overlapping a locus.

    ``None`` means at least one tile failed, so callers must not treat the
    variant as unconstrained.
    """
    elements: list[dict[str, float]] = []
    seen: set[tuple[int, int]] = set()
    pos = start
    tile = 50_000
    while pos <= end:
        chunk_end = min(end, pos + tile - 1)
        url = f"{ENSEMBL_REST}/overlap/region/human/{chrom}:{pos}-{chunk_end}"
        try:
            payload = client.get_json(
                url,
                params={"feature": "constrained"},
                headers={"Accept": "application/json"},
            )
        except HttpError as exc:
            logger.warning(
                "Constrained elements unavailable for %s:%s-%s (%s)", chrom, pos, chunk_end, exc
            )
            return None
        if not isinstance(payload, list):
            logger.warning("Unexpected constrained payload for %s:%s-%s", chrom, pos, chunk_end)
            return None
        for item in payload:
            key = (int(item["start"]), int(item["end"]))
            if key in seen:
                continue
            seen.add(key)
            score = item.get("score")
            elements.append(
                {
                    "start": key[0],
                    "end": key[1],
                    "score": float(score) if score not in (None, "") else 0.0,
                }
            )
        pos = chunk_end + 1
    return elements


def _fetch_region(client: HttpClient, chrom: str, start: int, end: int) -> str:
    """Fetch an inclusive interval. A failed window is split and retried."""
    start = max(1, int(start))
    end = max(start, int(end))
    try:
        return _fetch_exact(client, chrom, start, end)
    except HttpError:
        length = end - start + 1
        if length <= 400:
            raise
        mid = start + (length // 2) - 1
        logger.warning("Splitting failed Ensembl window %s:%s-%s", chrom, start, end)
        left = _fetch_region(client, chrom, start, mid)
        right = _fetch_region(client, chrom, mid + 1, end)
        return left + right


def _fetch_exact(client: HttpClient, chrom: str, start: int, end: int) -> str:
    """Fetch one inclusive window and check that the returned length matches."""
    url = f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{start}..{end}:1"
    payload = client.get_json(
        url,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    sequence = str(payload["seq"]).upper()
    expected = end - start + 1
    if len(sequence) != expected:
        raise ValueError(
            f"Ensembl returned {len(sequence)} bp for {chrom}:{start}-{end}, expected {expected}"
        )
    return sequence

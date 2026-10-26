"""Sequence utilities: reverse complement, local composition, interval merging."""

from __future__ import annotations

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def complement(base: str) -> str:
    """Return the DNA complement of a single base or a short string."""
    return base.translate(_COMPLEMENT).upper()


def revcomp(seq: str) -> str:
    """Reverse complement of a DNA string. Unknown bases other than N are kept."""
    return complement(seq)[::-1]


def gc_content(seq: str) -> float:
    """Fraction of A/C/G/T bases that are G or C. Empty input returns NaN."""
    bases = [base for base in seq.upper() if base in "ACGT"]
    if not bases:
        return float("nan")
    return sum(base in "GC" for base in bases) / len(bases)


def cpg_count(seq: str) -> int:
    """Count CG dinucleotides on the given strand."""
    text = seq.upper()
    return sum(1 for index in range(len(text) - 1) if text[index : index + 2] == "CG")


def apply_snv(seq: str, index: int, ref: str, alt: str) -> str:
    """Replace ``seq[index]`` after checking that it matches ``ref``."""
    if index < 0 or index >= len(seq):
        raise IndexError(index)
    if seq[index].upper() != ref.upper():
        raise ValueError(f"Expected {ref} at index {index}, found {seq[index]}")
    if len(alt) != 1:
        raise ValueError("apply_snv expects a single-base alt allele")
    return seq[:index] + alt.upper() + seq[index + 1 :]


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge inclusive integer intervals that touch or overlap."""
    if not intervals:
        return []
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def covering_tiles(intervals: list[tuple[int, int]], tile_bp: int) -> list[tuple[int, int]]:
    """Return the unique 1-based tiles of ``tile_bp`` that cover every interval.

    Tiles are aligned to a fixed grid (1..tile, tile+1..2*tile, ...) so two
    nearby variants share one cached Ensembl request.
    """
    if tile_bp < 1:
        raise ValueError("tile_bp must be positive")
    tiles: set[tuple[int, int]] = set()
    for raw_start, raw_end in intervals:
        start = max(1, int(min(raw_start, raw_end)))
        end = max(start, int(max(raw_start, raw_end)))
        first = (start - 1) // tile_bp
        last = (end - 1) // tile_bp
        for index in range(first, last + 1):
            tile_start = index * tile_bp + 1
            tiles.add((tile_start, tile_start + tile_bp - 1))
    return sorted(tiles)


class BaseStore:
    """Random access to a few genomic blocks fetched for one gene.

    Coordinates are 1-based and inclusive, matching Ensembl and ClinVar.
    """

    def __init__(self, blocks: list[tuple[int, str]]) -> None:
        cleaned: list[tuple[int, str]] = []
        for start, seq in blocks:
            text = seq.upper()
            if not text:
                continue
            cleaned.append((int(start), text))
        cleaned.sort(key=lambda item: item[0])
        self.blocks = cleaned

    def get(self, pos: int) -> str:
        """Return the base at a 1-based genomic position."""
        for start, seq in self.blocks:
            end = start + len(seq) - 1
            if start <= pos <= end:
                return seq[pos - start]
        raise KeyError(pos)

    def slice(self, start: int, end: int) -> str:
        """Return the inclusive sequence from ``start`` to ``end``."""
        if end < start:
            raise ValueError("end before start")
        for block_start, seq in self.blocks:
            block_end = block_start + len(seq) - 1
            if block_start <= start and end <= block_end:
                return seq[start - block_start : end - block_start + 1]
        raise KeyError((start, end))

    def __contains__(self, pos: int) -> bool:
        try:
            self.get(pos)
        except KeyError:
            return False
        return True

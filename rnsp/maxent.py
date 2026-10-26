"""MaxEntScan donor and acceptor scores.

The numeric models and the scoring algebra are the published MaxEntScan
matrices (Yeo and Burge, *J Comput Biol* 2004), as redistributed by the
MIT-licensed ``maxentpy`` project. See ``rnsp/data/maxent/NOTICE.md``.

A 9-mer donor uses exon positions -3,-2,-1 and intron positions +1..+6
(``cagGTAAGT`` is the usual consensus example). A 23-mer acceptor uses intron
positions -20..-1 and exon positions +1..+3. Scores are log2 odds. They are
not AlphaGenome outputs and they are not a position-weight matrix: the
acceptor model multiplies overlapping subsequence probabilities so that
nearby bases are not treated as independent.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

from rnsp.config import MAXENT_DIR

# Background and nearly-invariant dinucleotide frequencies shipped with MaxEntScan.
_DONOR_BACKGROUND = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
_DONOR_PLUS1 = {"A": 0.004, "C": 0.0032, "G": 0.9896, "T": 0.0032}
_DONOR_PLUS2 = {"A": 0.0034, "C": 0.0039, "G": 0.0042, "T": 0.9884}
_ACCEPTOR_BACKGROUND = {"A": 0.27, "C": 0.23, "G": 0.23, "T": 0.27}
_ACCEPTOR_MINUS2 = {"A": 0.9903, "C": 0.0032, "G": 0.0034, "T": 0.0030}
_ACCEPTOR_MINUS1 = {"A": 0.0027, "C": 0.0037, "G": 0.9905, "T": 0.0030}


def score_donor(seq: str, matrix: dict[str, float] | None = None) -> float:
    """MaxEntScan 5' splice-site score for a 9-mer on the coding strand."""
    motif = seq.upper()
    if len(motif) != 9 or any(base not in "ACGT" for base in motif):
        raise ValueError("donor score expects a 9-mer of ACGT")
    table = matrix if matrix is not None else load_donor_matrix()
    dinucleotide = motif[3:5]
    dinucleotide_term = (
        _DONOR_PLUS1[dinucleotide[0]]
        * _DONOR_PLUS2[dinucleotide[1]]
        / (_DONOR_BACKGROUND[dinucleotide[0]] * _DONOR_BACKGROUND[dinucleotide[1]])
    )
    rest = motif[:3] + motif[5:]
    return math.log(dinucleotide_term * table[rest], 2)


def score_acceptor(seq: str, matrix: dict[int, dict[int, float]] | None = None) -> float:
    """MaxEntScan 3' splice-site score for a 23-mer on the coding strand."""
    motif = seq.upper()
    if len(motif) != 23 or any(base not in "ACGT" for base in motif):
        raise ValueError("acceptor score expects a 23-mer of ACGT")
    table = matrix if matrix is not None else load_acceptor_matrix()
    dinucleotide = motif[18:20]
    dinucleotide_term = (
        _ACCEPTOR_MINUS2[dinucleotide[0]]
        * _ACCEPTOR_MINUS1[dinucleotide[1]]
        / (_ACCEPTOR_BACKGROUND[dinucleotide[0]] * _ACCEPTOR_BACKGROUND[dinucleotide[1]])
    )
    rest = motif[:18] + motif[20:]
    rest_score = 1.0
    rest_score *= table[0][_hashseq(rest[:7])]
    rest_score *= table[1][_hashseq(rest[7:14])]
    rest_score *= table[2][_hashseq(rest[14:])]
    rest_score *= table[3][_hashseq(rest[4:11])]
    rest_score *= table[4][_hashseq(rest[11:18])]
    rest_score /= table[5][_hashseq(rest[4:7])]
    rest_score /= table[6][_hashseq(rest[7:11])]
    rest_score /= table[7][_hashseq(rest[11:14])]
    rest_score /= table[8][_hashseq(rest[14:18])]
    return math.log(dinucleotide_term * rest_score, 2)


@functools.lru_cache(maxsize=1)
def load_donor_matrix(path: str | None = None) -> dict[str, float]:
    """Load the 7-mer (non-GT) donor probability table."""
    matrix_path = Path(path) if path else MAXENT_DIR / "score5_matrix.txt"
    matrix: dict[str, float] = {}
    with matrix_path.open(encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split()
            matrix[key.upper()] = float(value)
    return matrix


@functools.lru_cache(maxsize=1)
def load_acceptor_matrix(path: str | None = None) -> dict[int, dict[int, float]]:
    """Load the nine overlapping acceptor subsequence tables."""
    matrix_path = Path(path) if path else MAXENT_DIR / "score3_matrix.txt"
    matrix: dict[int, dict[int, float]] = {}
    with matrix_path.open(encoding="utf-8") as handle:
        for line in handle:
            table_id, key, value = line.split()
            matrix.setdefault(int(table_id), {})[int(key)] = float(value)
    return matrix


def _hashseq(seq: str) -> int:
    """Pack an ACGT string into the integer key used by the acceptor matrices."""
    code = {"A": 0, "C": 1, "G": 2, "T": 3}
    packed = 0
    for base in seq:
        packed = packed * 4 + code[base]
    return packed

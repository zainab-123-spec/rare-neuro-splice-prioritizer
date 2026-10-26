"""Command line for dataset building, evaluation, and single-gene reports."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from rnsp import __version__
from rnsp.config import FIGURE_DIR, PROCESSED_DIR, REPORT_DIR
from rnsp.dataset import build_dataset
from rnsp.model import evaluate, load_metrics, save_evaluation
from rnsp.plots import markdown_table, write_figures
from rnsp.report import render_benchmark_html, render_gene_html


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run one subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    if args.command == "build":
        symbols = _split_genes(args.genes)
        build_dataset(symbols, out_dir=args.outdir)
        return 0
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "run":
        symbols = _split_genes(args.genes)
        build_dataset(symbols, out_dir=args.outdir)
        return _evaluate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser. Tests import this directly."""
    parser = argparse.ArgumentParser(
        prog="rnsp",
        description=(
            "Prioritize non-coding splice variants in rare neurological disease genes "
            "using ClinVar, Ensembl, GTEx, and interpretable splicing features."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rnsp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Download public data and write the variant table")
    _add_shared(build)

    evaluate_parser = sub.add_parser("evaluate", help="Cross-validate models and write figures")
    _add_shared(evaluate_parser)
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.add_argument("--splits", type=int, default=5)

    analyze = sub.add_parser("analyze", help="Write a one-gene figure and HTML report")
    analyze.add_argument("gene", help="Panel gene symbol, for example SCN1A")
    _add_shared(analyze)

    run = sub.add_parser("run", help="Build the dataset, then evaluate it")
    _add_shared(run)
    run.add_argument("--bootstrap", type=int, default=1000)
    run.add_argument("--splits", type=int, default=5)
    return parser


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--genes",
        default="",
        help="Comma-separated panel symbols. Default: the full 30-gene panel.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for tables, metrics, and the model.",
    )


def _split_genes(text: str) -> list[str] | None:
    symbols = [item.strip() for item in text.split(",") if item.strip()]
    return symbols or None


def _evaluate(args: argparse.Namespace) -> int:
    variants_path = Path(args.outdir) / "variants.csv"
    if not variants_path.exists():
        print(f"Missing {variants_path}. Run: python -m rnsp build", file=sys.stderr)
        return 1
    frame = pd.read_csv(variants_path, dtype={"chrom": str, "gene": str})
    metrics, scored, estimator = evaluate(frame, n_splits=args.splits, n_bootstrap=args.bootstrap)
    save_evaluation(metrics, scored, estimator, out_dir=args.outdir)
    structures_path = Path(args.outdir) / "gene_structures.json"
    structures = {}
    if structures_path.exists():
        structures = json.loads(structures_path.read_text(encoding="utf-8"))
    gtex_path = Path(args.outdir) / "gtex_median_tpm.csv"
    gtex = pd.read_csv(gtex_path) if gtex_path.exists() else None
    write_figures(scored, metrics, gtex, structures, FIGURE_DIR)
    qc_path = Path(args.outdir) / "qc_report.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8")) if qc_path.exists() else None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "benchmark.html").write_text(render_benchmark_html(metrics, qc), encoding="utf-8")
    (FIGURE_DIR / "metrics_table.md").write_text(markdown_table(metrics), encoding="utf-8")
    print(markdown_table(load_metrics(Path(args.outdir) / "metrics.json")))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    gene = args.gene.strip().upper()
    scored_path = Path(args.outdir) / "variants_scored.csv"
    plain_path = Path(args.outdir) / "variants.csv"
    path = scored_path if scored_path.exists() else plain_path
    if not path.exists():
        print(
            f"Missing variant table in {args.outdir}. Run build and evaluate first.",
            file=sys.stderr,
        )
        return 1
    frame = pd.read_csv(path, dtype={"chrom": str, "gene": str})
    structures_path = Path(args.outdir) / "gene_structures.json"
    if not structures_path.exists():
        print(f"Missing {structures_path}.", file=sys.stderr)
        return 1
    structures = json.loads(structures_path.read_text(encoding="utf-8"))
    if gene not in structures or frame[frame["gene"] == gene].empty:
        print(f"No QC-passed variants for {gene}.", file=sys.stderr)
        return 1
    from rnsp.plots import plot_gene_architecture

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figure = FIGURE_DIR / f"{gene}_architecture.png"
    plot_gene_architecture(frame[frame["gene"] == gene], structures[gene], figure)
    report = REPORT_DIR / f"{gene}.html"
    report.write_text(render_gene_html(gene, frame, structures[gene]), encoding="utf-8")
    part = frame[frame["gene"] == gene]
    print(
        f"{gene}: {len(part)} variants "
        f"({int(part['label'].sum())} pathogenic / likely pathogenic). "
        f"Figure {figure}. Report {report}."
    )
    return 0

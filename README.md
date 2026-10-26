# Rare neurological splice-variant prioritizer

[![Tests](https://github.com/zainab-123-spec/rare-neuro-splice-prioritizer/actions/workflows/tests.yml/badge.svg)](https://github.com/zainab-123-spec/rare-neuro-splice-prioritizer/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Interpretable regulatory-genomics model for **non-coding RNA splicing variants** in rare Mendelian neurological disease. It asks a narrower version of the question behind Google DeepMind's [AlphaGenome](https://doi.org/10.1038/s41586-025-10014-0): when a variant sits in the 98% of the genome that does not code for protein, can we tell a splice-disrupting allele from a benign one, and does that signal still work on a **gene the model has never seen**?

This repository does not reimplement AlphaGenome and it does not call the AlphaGenome API. AlphaGenome is a long-context deep model that predicts splice-site usage, splice-junction strength, expression, and chromatin tracks from a megabase of DNA. The project here is the interpretable baseline a reviewer should demand first: distance to the canonical junction, the GT/AG dinucleotide rule, a position-weight matrix, MaxEntScan deltas, polypyrimidine and branchpoint changes, local GC/CpG, Ensembl GERP-constrained elements, and brain expression from GTEx and the Human Protein Atlas.

**Author:** Zainab Javed, BS Bioinformatics, Quaid-i-Azam University, Islamabad. GitHub [zainab-123-spec](https://github.com/zainab-123-spec).

> Research and education tool only. This is not a medical device and it does not provide a diagnosis or a clinical classification. ClinVar labels are submitted assertions, not proof that a variant acts through splicing.

## Benchmark

Numbers below are out-of-fold measurements from `python -m rnsp evaluate` on the QC-passed ClinVar cohort. They are not training-set scores. Intervals are percentile bootstrap 95% CIs on the pooled out-of-fold predictions. A cell is omitted when a slice does not have enough variants of both classes.

<!-- BENCHMARK_TABLE_START -->
The measured comparison is written to [`docs/figures/metrics_table.md`](docs/figures/metrics_table.md) when you run the evaluator. Cohort size, class balance, and the table itself are filled from that run so the README cannot drift from `data/processed/metrics.json`.
<!-- BENCHMARK_TABLE_END -->

### How to read the comparison

Two splits are reported on purpose.

- **Gene held out** (`GroupKFold`, 5 folds). Every variant from a gene is in the same fold. The test gene is absent from training. This is the number that speaks to a new neurological disease gene.
- **Random split** (`StratifiedKFold`, shuffled, 5 folds). Variants from the same gene appear on both sides. This number is usually optimistic, because ClinVar ascertainment is gene-specific.

Two fixed rules are the baselines, and they are not trained:

- **Distance rule.** Score = `1 / (1 + |distance to the nearest canonical exon–intron junction|)`.
- **Canonical ±1/±2 rule.** Score = 1 when the SNV changes the intronic GT (donor +1/+2) or AG (acceptor −1/−2), else 0.

The hard slice is **non-canonical variants at ±3 to ±50 bp**. The dinucleotide rule is constant there, so its ROC-AUC is 0.5 whenever both classes are present. Models in that slice are retrained inside the slice (`train_on_subset`) so canonical ±1/±2 sites cannot carry the fit.

Figures, all produced by the same run:

| Figure | What it shows |
| --- | --- |
| [ROC curves](docs/figures/roc_curves.png) | Gene-held-out versus random split |
| [Precision–recall](docs/figures/pr_curves.png) | Same models on a prevalence-aware curve |
| [Metric bars](docs/figures/metrics_comparison.png) | ROC-AUC, PR-AUC, and Brier score |
| [Hard subset](docs/figures/hard_subset_metrics.png) | ±3..±50 bp only |
| [Calibration](docs/figures/calibration.png) | Gene-held-out probabilities |
| [Coefficients](docs/figures/logistic_coefficients.png) | Standardized logistic weights, full-cohort refit |
| [Permutation importance](docs/figures/permutation_importance.png) | Held-out AUC drop, HistGradientBoosting |
| [Distance](docs/figures/distance_by_label.png) | Why a distance rule looks strong |
| [Cohort](docs/figures/cohort_by_gene.png) | Pathogenic and benign counts per gene |
| [Brain expression](docs/figures/brain_expression.png) | GTEx v8 median TPM |
| [SCN1A map](docs/figures/SCN1A_architecture.png) | Canonical transcript with ClinVar variants |

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m rnsp build          # ClinVar + Ensembl + GTEx + HPA, with a disk cache
python -m rnsp evaluate       # cross-validation, figures, HTML benchmark
python -m rnsp analyze SCN1A  # one-gene map and ranked table
streamlit run app.py
```

`build` respects NCBI's limit of 3 requests/second without an API key and caches every successful response under `data/cache/` (gitignored). A second run reuses the cache. Processed tables, the PWM, metrics, and the deployment model are written to `data/processed/`.

Useful narrower commands:

```bash
python -m rnsp build --genes SCN1A,STXBP1 --outdir /tmp/rnsp-smoke
python -m rnsp evaluate --bootstrap 200
```

## What is computed for each variant

All motif math is done on the **coding strand**. Minus-strand genes are reverse-complemented, and the genomic alternate allele is complemented with them. The reference base must match the Ensembl sequence or the variant is dropped.

| Feature | Definition |
| --- | --- |
| `abs_dist_to_splice` | Distance to the nearest canonical exon–intron boundary. The adjacent base on either side is distance 1. |
| `disrupts_canonical_dinucleotide` | 1 if the SNV sits on donor +1/+2 or acceptor −1/−2. |
| `donor_maxent_delta`, `acceptor_maxent_delta` | MaxEntScan log-odds (Yeo & Burge 2004), alternate minus reference, when the variant overlaps the 9-mer (−3..+6) or 23-mer (−20..+3). |
| `donor_pwm_delta`, `acceptor_pwm_delta` | Same windows, independent-base information content. The PWM is fit on the panel's own canonical junctions with a 0.5 pseudocount and a 0.25 background. Labels are not used. |
| `cryptic_*_delta` | Change in the best MaxEnt score of a non-annotated site in ±80 bp. |
| `cryptic_*_created` | 1 when the alt allele creates a new GT or AG whose MaxEntScan score is at least 3. |
| `ppt_delta` | Change in pyrimidine fraction on acceptor positions −20..−3. |
| `branchpoint_delta` | Change in the best YTRAC match from −48 to −18. |
| `gc_delta`, `cpg_delta`, `gc_local` | GC fraction and CpG count in the Ensembl ±50 bp window. |
| `motif_ref_information_bits` | PWM information of the reference base that was changed. |
| `constraint_overlap`, `constraint_score` | Overlap with an Ensembl GERP constrained element. Missing calls stay missing; they are not filled with zero. |
| GTEx / HPA fields | Gene-level brain specificity, tau, cortex / hippocampus / cerebellum TPM, and HPA brain nTPM. |

Gene-level expression cannot separate two variants in the same gene. Under gene hold-out it can still proxy which genes are heavily submitted to ClinVar. The `motif` feature set drops distance, the dinucleotide flag, and every expression feature so that comparison is visible.

## Gene panel

Thirty genes, chosen for Mendelian neurological disease plus the classic splice-disease genes on the requested list. CFTR is cystic fibrosis. It is on the panel because splice donor and acceptor biology was worked out there, and because leaving it in makes the brain-expression feature honest: CFTR is not a brain gene.

| Category | Genes |
| --- | --- |
| Epilepsy and neurodevelopmental | SCN1A, DNM1, MECP2, STXBP1, KCNQ2, SYNGAP1, CDKL5, TSC1, TSC2, CHD2, SCN2A, SCN8A, GABRG2, CACNA1A |
| Neuromuscular | DMD, SMN1 |
| Neurodegenerative | SOD1, TARDBP, FUS, LRRK2, PARK7, PINK1 |
| Neurometabolic | HEXA, PAH, GBA1, ATP7B, NPC1, GALC, ABCD1 |
| Classic splice benchmark | CFTR |

Labels are ClinVar germline classifications: pathogenic or likely pathogenic versus benign or likely benign. Conflicting and uncertain records are removed. Only GRCh38 single-nucleotide variants with a canonical SPDI allele are kept, and only when the molecular consequence is splice donor, splice acceptor, splice region, intron, or 5′/3′ UTR. A coding allele that is also a UTR variant on another transcript is dropped (`keep_consequence`), and UTR searches themselves exclude coding consequence phrases so those records do not fill the download cap. ClinVar's index does not contain the phrase "splice region variant" (the query count is zero). Non-canonical sites such as c.3879+5 are returned by the intron query. Each gene × label × consequence query is capped (see `rnsp.config.CONSEQUENCE_RETMAX`; intron is capped at 160) so one class cannot flood the cohort. The cap is a sampling limitation and is recorded in `data/processed/qc_report.json`.

## Public APIs

1. **NCBI ClinVar E-utilities** (`esearch`, `esummary`) for labels, consequences, review status, and GRCh38 SPDI alleles.
2. **Ensembl REST** for the canonical transcript and exons (`lookup`), the reference window (`sequence/region`), and GERP constrained elements (`overlap` `feature=constrained`).
3. **GTEx Portal API v2** `medianGeneExpression` for dataset `gtex_v8`, after `geneSearch` resolves a versioned GENCODE id. **Human Protein Atlas** gene JSON supplies brain nTPM and tissue-specificity labels.

There is no AlphaGenome dependency. Expression Atlas was not required once GTEx and HPA both returned data.

## Repository layout

```
rnsp/                 Python package
  clinvar.py          E-utilities search and the SPDI parser
  ensembl.py          Exons, sequence tiles, constrained elements
  expression.py       GTEx and Human Protein Atlas
  splicing.py         Junction geometry, PWM, cryptic / PPT / branchpoint
  maxent.py           MaxEntScan donor and acceptor scores
  features.py         One feature row per variant
  dataset.py          Build, QC, and write the tables
  model.py            GroupKFold, StratifiedKFold, baselines, Brier score
  plots.py            Figures
  report.py           HTML reports
  cli.py              python -m rnsp
app.py                Streamlit explorer
tests/                Pytest suite, no network
data/processed/       Cohort, metrics, model (after a run)
docs/figures/         PNG figures
docs/reports/         HTML reports
LEARNING_GUIDE.md     Biology, modules, and interview questions
```

The MaxEntScan matrices under `rnsp/data/maxent/` are third-party data. Attribution and the MIT license are in that directory.

## Tests and CI

```bash
ruff check . && ruff format --check .
pytest
```

GitHub Actions runs both on Python 3.10 and 3.12 (`.github/workflows/tests.yml`). Tests use a real ClinVar esummary fixture and the published MaxEntScan example scores (`cagGTAAGT` → 10.86). They do not call NCBI or Ensembl.

## Limitations

- ClinVar is not a random sample of variation. Pathogenic splice sites are submitted because a patient was sequenced. Benign intron variants are submitted because a panel found them. Distance to the junction will look strong for that reason alone. The hard subset is the check on that story.
- The canonical transcript is Ensembl's, which may differ from the RefSeq transcript named in an HGVS string. Distance is genomic distance to the Ensembl canonical junctions.
- The PWM assumes independent bases. MaxEntScan does not. Neither model is AlphaGenome, SpliceAI, or a full thermodynamic U1/U2 calculation.
- Only SNVs are scored. Indels, especially the SMN1 exon 7 events, are out of scope.
- Query caps truncate very large ClinVar slices. The retained IDs follow E-utilities' default order. The splice-region consequence phrase is absent from the ClinVar index, so that slice is empty by construction and those alleles are recovered as intron variants.
- The deployment score in the app is a refit on the whole research cohort. Quote the cross-validated table, not that score.
- Nothing here is cleared for diagnostic use.

## References

- Avsec Ž, Latysheva N, Cheng J, et al. Advancing regulatory variant effect prediction with AlphaGenome. *Nature*. 2026;649:1206–1218. doi:[10.1038/s41586-025-10014-0](https://doi.org/10.1038/s41586-025-10014-0).
- Yeo G, Burge CB. Maximum entropy modeling of short sequence motifs with applications to RNA splicing signals. *Journal of Computational Biology*. 2004;11:377–394.
- Jaganathan K, et al. Predicting splicing from primary sequence with deep learning. *Cell*. 2019;176:535–548. (SpliceAI, the deep-learning splice baseline this project does not try to replace.)
- Landrum MJ, et al. ClinVar: improvements to accessing data. *Nucleic Acids Research*. 2020.
- GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*. 2020;369:1318–1330.
- Uhlén M, et al. Tissue-based map of the human proteome. *Science*. 2015;347:1260419. (Human Protein Atlas.)

# Learning guide

This guide explains the biology, then the code, then fifteen interview questions you should be able to answer from this repository. It is written in plain English. No step assumes you already work in a splicing lab.

The project lives at [github.com/zainab-123-spec/rare-neuro-splice-prioritizer](https://github.com/zainab-123-spec/rare-neuro-splice-prioritizer).

## 1. The biology, without jargon first

A gene is a stretch of DNA. The cell copies it into RNA, then cuts that RNA. The pieces that stay in the mature messenger RNA are **exons**. The pieces that are removed are **introns**. The cut-and-paste step is **splicing**.

Most disease-causing variants that people learn first change a protein letter (missense) or stop the protein early (nonsense). Those variants sit in the roughly 2% of the genome that codes for protein. The other 98% is non-coding. A lot of it is intron. Variants there can still cause disease if they make the cell cut the RNA in the wrong place.

Two short signals mark a typical human intron:

- The **donor**, or 5′ splice site, at the start of the intron. The first two intron bases are almost always **GT**. The usual window people score is three exon bases, then six intron bases: positions −3, −2, −1, +1, +2, +3, +4, +5, +6. The consensus looks like `CAG|GTAAGT`.
- The **acceptor**, or 3′ splice site, at the end of the intron. The last two intron bases are almost always **AG**. Upstream of that AG is a **polypyrimidine tract**, a run of C and T (the pyrimidines, letter Y). Further upstream, often around −18 to −40, is a **branchpoint**. A short form of the branchpoint motif is **YTRAC** (Y = C or T, R = A or G).

The clinical rule of thumb is: a change at the donor +1 or +2, or the acceptor −1 or −2, is very likely to break splicing. That is the **canonical dinucleotide**. It is not the whole story. A change at donor +5, or a deep intronic variant that creates a new GT, can also be pathogenic. Those are the variants where a one-line rule fails and a model has something to prove.

Neurological and neurodevelopmental genes are a useful place to study this. Transcripts in the brain are long, alternative splicing is busy, and rare Mendelian diseases (Dravet syndrome and *SCN1A*, Rett syndrome and *MECP2*, Duchenne muscular dystrophy and *DMD*, spinal muscular atrophy and *SMN1*) include famous splicing mechanisms. This project stays away from cancer. Somatic tumor variant interpretation is a different problem.

## 2. What AlphaGenome has to do with this

In 2026, Google DeepMind published **AlphaGenome** (Avsec et al., *Nature* 649:1206–1218). It reads about a megabase of DNA and predicts many experimental tracks at single-base resolution: expression, chromatin, transcription-factor binding, and splicing (which bases are used as donors or acceptors, and how strong the junctions are). It is a deep model. It is good at exactly the problem this repository is about: non-coding variant effect, including splicing.

This code is not a small AlphaGenome. A student project that claimed that would be misleading. What you can honestly say in an interview is:

- AlphaGenome shows that sequence alone carries splice and regulatory signal.
- Before trusting a giant model, you want simple features whose failures you understand.
- Those features are the distance to the real junction, the GT/AG rule, a weight matrix, a maximum-entropy splice score, and the tract and branchpoint signals.
- You also want a test that does not leak the gene. Holding out the gene is that test.

SpliceAI (Jaganathan et al., *Cell* 2019) is the earlier deep-learning splice model people will mention. Same answer: this project is the interpretable baseline, not a replacement.

## 3. Where the data come from

Three public REST APIs, plus the Human Protein Atlas next to GTEx.

**ClinVar, through NCBI E-utilities.** ClinVar is a public archive of variant classifications submitted by labs. We search `db=clinvar` with `esearch`, then download records with `esummary`. We keep pathogenic and likely pathogenic versus benign and likely benign. We drop uncertain and conflicting labels. We keep splice donor, splice acceptor, splice region, intron, and UTR consequences. A nonsense or missense allele can also be labeled "5 prime UTR variant" on a second transcript; those mixed records are excluded, and the UTR query itself adds `NOT` clauses for coding consequences. ClinVar does not index the words "splice region variant", so c.+3 to c.+50 alleles arrive as intron variants. The genomic allele comes from the **SPDI** string, which is 0-based, so a substitution at SPDI position `P` is VCF/Ensembl position `P+1`.

**Ensembl REST.** For each gene we look up the canonical transcript and its exons. We download the reference DNA in short tiles (`sequence/region/human/...`) and check that the ClinVar reference base is actually on the chromosome. We also request **constrained** elements, which are GERP conserved stretches. If that call fails, the constraint features stay missing. They are not silently set to zero, because "we do not know" is not the same as "this base is unconstrained."

**GTEx Portal API v2.** Median transcripts-per-million (TPM) for 54 tissues in the `gtex_v8` dataset, including cortex, hippocampus, and cerebellum. **Human Protein Atlas** adds a brain nTPM value and a tissue-specificity label. These numbers are properties of the gene, not of the variant.

Every successful HTTP response is stored under `data/cache/` as a content-addressed file. NCBI is limited to about three requests per second when you do not have an API key. The client sleeps, retries 503 and 429 responses, and does not cache failures.

## 4. A walk through the package

Read the files in this order.

### `rnsp/config.py`

The gene panel, the API URLs, the feature names, and the rate limits. Each gene has a category and a disease name so plots and reports can group Dravet syndrome separately from cystic fibrosis. `GBA1` also searches the alias `GBA`, because symbol history in ClinVar is messy, but a record is kept only when the gene list actually contains the panel symbol or its alias.

### `rnsp/http.py`

`RateLimiter` and `HttpClient`. The cache key is a hash of the method, URL, and query parameters. Tests pass a fake session, so CI never touches the network.

### `rnsp/sequence.py`

Reverse complement, GC content, CpG count, and `BaseStore`, which holds only the genomic tiles we fetched. Coordinates are 1-based and inclusive, same as Ensembl and ClinVar.

### `rnsp/clinvar.py`

Builds the search term, parses esummary JSON, and turns a record into a `ClinVarVariant` or a drop reason. `parse_spdi` accepts only single-base substitutions. The GRCh38 `variation_loc` start must equal the SPDI position plus one. Duplicate alleles collapse. If the same allele is both pathogenic and benign, both copies are dropped.

### `rnsp/ensembl.py`

`fetch_gene` reads the canonical transcript. `fetch_sequence_store` merges windows around every exon edge and every variant, then downloads those windows in 8 kb tiles. `fetch_constrained` walks the locus in 50 kb chunks.

### `rnsp/expression.py`

`tau_specificity` is Yanai's tau. Zero means the gene is expressed everywhere at the same level. One means it is specific. `gtex_brain_specificity` is mean brain TPM divided by mean TPM across all returned tissues. A value above 1 means the brain average is higher than the tissue average.

### `rnsp/splicing.py`

This is the biology module.

Exons from Ensembl always have `start <= end`, even on the minus strand. Transcription direction is the strand. On the plus strand the donor +1 base is `exon_end + 1`. On the minus strand it is `exon_start - 1`, because transcription walks toward smaller coordinates. The acceptor −1 base is the mirror of that.

Distance counts the adjacent base as 1. Intronic donor +1 is distance 1. The last exonic base is also distance 1, but it is not the canonical dinucleotide, because the canonical dinucleotide is the intron side. Region classes:

- `canonical_dinucleotide`: intronic distance 1 or 2
- `hard_3_50`: distance 3 through 50, not the canonical dinucleotide
- `deep_gt_50`: farther than 50
- `other_near`: the exonic flank at distance 1 or 2

The PWM is a column of base frequencies. The score of a motif is the sum of `log2(frequency / 0.25)` across positions, in bits. It assumes positions are independent. We fit it on the reference junctions of the panel after sequences are downloaded, with pseudocount 0.5, and we require at least 15 junctions. The ClinVar label is not an input.

### `rnsp/maxent.py`

MaxEntScan (Yeo and Burge, 2004) gives a log-odds score for a 9-mer donor or a 23-mer acceptor. The acceptor model multiplies overlapping subsequence probabilities, so neighboring bases are not independent. The numeric tables are the published ones, vendored with attribution. The unit tests lock the well-known examples: `cagGTAAGT` scores about 10.86, and a weak donor `taaATAAGT` scores about −0.12.

Delta means alternate score minus reference score. A large negative donor delta means the variant weakened the annotated donor.

### `rnsp/features.py`

One function, `compute_features`, builds the row the model sees. Cryptic sites are MaxEnt scores in a ±80 bp coding-strand window, ignoring the annotated junction. A cryptic alert fires when the alternate allele creates a GT or AG that was not there and the new site scores at least 3.

### `rnsp/dataset.py`

The orchestration. For each gene: search ClinVar, parse, drop bad coordinates, download Ensembl sequence and constraint, download expression. Then fit one PWM on all canonical junctions. Then score every variant. Reference mismatches are counted, not patched. The outputs are `variants.csv`, `qc_report.json`, `pwm.json`, `gene_structures.json`, and `gtex_median_tpm.csv`.

### `rnsp/model.py`

Three classifiers from scikit-learn:

- L2 logistic regression, balanced class weights, median imputation, standardized features
- `HistGradientBoostingClassifier`
- `GradientBoostingClassifier`

Trees receive balanced sample weights computed inside each training fold only.

`GroupKFold` splits by gene. `StratifiedKFold` splits at random but keeps the class fraction. Predictions are **out of fold**: each variant is scored by a model that did not train on it. ROC-AUC, average precision (PR-AUC), and Brier score are computed on those predictions. Bootstrap intervals resample the out-of-fold pairs.

The distance rule and the dinucleotide rule are not fit. A one-feature logistic regression on distance is also reported, so Brier score for "distance alone" has a fitted probability, separate from the fixed `1/(1+distance)` rule.

On the hard slice we do two things. We score the all-data model on the hard variants (`train_all_eval_subset`). We also refit inside the slice (`train_on_subset`). The second number is the one to quote when someone asks "what happens when the easy GT/AG sites are gone."

A full-cohort refit is saved as `model.joblib` for the explorer. Its score is labeled `deployment_score`. It is not the benchmark.

### `rnsp/plots.py` and `rnsp/report.py`

Matplotlib figures and self-contained HTML. The clinical disclaimer is in the HTML. Gene symbols are escaped so a label cannot inject markup.

### `rnsp/cli.py` and `app.py`

`python -m rnsp build`, `evaluate`, `analyze SCN1A`, and `run`. The Streamlit app reads the processed files. It draws the exon map in transcript order (introns compressed, otherwise a 200 kb gene hides its exons) and a bar chart of motif deltas for one variant.

## 5. How to run it yourself

```bash
python -m pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
python -m rnsp build
python -m rnsp evaluate
python -m rnsp analyze SCN1A
streamlit run app.py
```

`pytest` does not need a network. `build` does. Expect the first build to take a while, because ClinVar is rate-limited and Ensembl occasionally returns 503. The cache makes the second build much shorter.

## 6. How to read the main figure

Open `docs/figures/roc_curves.png`. The left panel holds out genes. The right panel is a random split. The diagonal is chance. A curve that bows toward the top left is better. If the right panel is clearly above the left, the random split was flattering and you should quote the left panel.

Then open `docs/figures/hard_subset_metrics.png`. On the measured cohort the ±1/±2 rule sits at ROC-AUC 0.500 inside ±3..±50 bp, the distance rule falls to 0.590, and HistGradientBoosting using only motif features reaches 0.931 (PR-AUC 0.784) when the test genes are held out. The full-cohort gene-held-out gradient boosting ROC-AUC is 0.975, and that number is easier than the hard slice because almost every canonical dinucleotide in the download is pathogenic (1,544 versus 4 benign). Quote the hard slice when someone asks whether the model does more than memorize GT/AG.

Brier score is the mean squared error of the predicted probability. Lower is better. A model can rank well (high AUC) and still be badly calibrated (high Brier) if its probabilities are overconfident. The calibration plot checks that.

## 7. Quality control you should be able to defend

A variant is dropped when:

- the gene symbol is not on the ClinVar record
- the label is uncertain, conflicting, or something other than the four kept classes
- the consequence is not splice, intron, or UTR
- there is no GRCh38 single-nucleotide SPDI that matches the location
- the chromosome disagrees with the Ensembl gene
- the position is far outside the locus
- the Ensembl base is not the ClinVar reference base
- the same allele has two labels

`data/processed/qc_report.json` counts each reason. If reference mismatches dominate, the coordinate code is wrong. Do not "fix" that by skipping the check.

## 8. Fifteen interview questions

These are questions this project prepares you to answer. They are not a dump of anyone's internal interview bank.

### 1. Why hold out the gene instead of splitting variants at random?

Because the scientific claim is "this works on a neurological gene we did not train on." A random split puts other variants from *SCN1A* in the training fold. The model can learn that *SCN1A* submissions are mostly pathogenic splice sites, which will not help on a new gene. `GroupKFold` on the gene id stops that leak. You should still report the random split, because the gap between the two numbers is itself the result.

### 2. Why is ROC-AUC not enough?

ROC-AUC treats false positives and false negatives as a ranking problem. It can look excellent when the positive class is rare and you only care about the top of the list, or when a trivial rule separates most of the mass. PR-AUC (average precision) watches precision as you move down the ranked list, so prevalence is visible. Brier score asks whether a predicted 0.9 is right about 90% of the time. A prioritizer that says 0.99 for everything it ranks first is not usable, even if the ranking is perfect. Report all three.

### 3. What does the canonical dinucleotide rule miss?

Any real splice effect that is not a change of GT or AG at the intron edge. Donor +5 is the textbook example: it is outside ±1/±2, it is highly conserved, and MaxEntScan or a PWM will usually drop. Deep intronic variants that create a new donor, and changes to the polypyrimidine tract, are the same kind of miss. On the ±3..±50 slice the rule outputs the same score for every variant, so ROC-AUC falls to 0.5.

### 4. What is the difference between a PWM and MaxEntScan?

A position-weight matrix scores each base on its own and adds the information. It cannot say "a G at +5 hurts more when +6 is already a mismatch," because it never looks at pairs. MaxEntScan's acceptor model multiplies probabilities of overlapping chunks, which is a way of keeping those dependencies without listing every 23-mer. Both are used here, on the same windows (−3..+6 and −20..+3), so you can see whether the dependency model changes the delta. Neither is a neural net.

### 5. How do you score a gene on the minus strand?

Ensembl stores exons with the smaller coordinate first, regardless of strand. The DNA sequence from `sequence/region` with strand 1 is the plus-strand reference, which is also what ClinVar's SPDI describes. To score a motif you walk the offsets in transcription order and complement each base when the gene is on the minus strand. A coding G>A at donor +1 becomes a genomic C>T. Forgetting the complement will make every minus-strand reference check fail, or it will score the wrong motif. The unit test `test_minus_strand_genomic_allele_is_complemented` locks this.

### 6. What is leakage in this dataset, concretely?

Using the ClinVar molecular-consequence string as a feature would leak, because "splice donor variant" is almost a restatement of the label-generating process. Using the review-star rating can leak, because pathogenic submissions often have more review. Using the gene id as a feature defeats the hold-out. Expression is subtler: it is legitimate at prediction time (you know the gene's TPM) but it is constant inside a gene, so a model can use "this gene is brain-specific and its submissions are pathogenic." That is why the motif-only feature set exists.

### 7. Why can a distance baseline look better than a careful splicing model?

Because of how ClinVar is collected. Pathogenic splice variants cluster on the GT/AG. Benign intronic variants are often far away, or they would not have been called benign by a simple rule. Distance reconstructs that sampling. It is a real pattern. It is not evidence that distance is the mechanism for the variants that are not at the junction. Show the histogram (`distance_by_label.png`) and then the hard-subset table in the same breath.

### 8. How does the HTTP client avoid hammering NCBI, and how do you test it?

A `RateLimiter` records the last request time and sleeps until `min_interval` has passed. NCBI's unauthenticated limit is 3 per second, so the interval is 0.34 s. Ensembl and GTEx have their own clients and their own intervals, because the limits belong to the host, not to the program. Retries cover 429 and 5xx with exponential backoff. Only a body that parsed as JSON is cached. Tests inject a fake session that returns 503 once and then 200, and a clock that does not really sleep. The test asserts a 404 is not retried.

### 9. What is calibration, and why would a hiring committee care?

Calibration means the probabilities mean what they say. If you take every variant the model scored between 0.8 and 0.9, about 85% of them should be pathogenic. Ranking metrics ignore that. A software engineer shipping a score into a UI is shipping a probability whether they meant to or not. Brier score plus a reliability curve is the minimum check. In this project the curve is built from gene-held-out predictions, not from the full-cohort refit.

### 10. How would you explain GroupKFold to someone who has only used `train_test_split`?

`train_test_split` draws rows. If rows are variants and several rows share a gene, the gene is on both sides. GroupKFold takes an array of group labels and guarantees that each group is entirely in training or entirely in testing. With 30 genes and 5 folds, each test fold holds out about 6 genes. You still need both classes inside enough folds to compute AUC. If a fold's training data has one class, that fold is skipped rather than crashing.

### 11. What would you monitor if this were a small internal service?

The QC drop rate, especially reference mismatches, because a silent coordinate bug looks like "no variants today." The fraction of constraint and expression fields that are missing. The latency and 503 rate of Ensembl and NCBI. The gap between gene-held-out AUC and random-split AUC, so a "model improvement" that only helps the random split is visible. And the class balance per gene, so one gene with 2,000 benign introns cannot dominate a later retraining without you noticing.

### 12. Why median-impute and also keep a missingness flag?

Median imputation lets logistic regression and the older gradient-boosting implementation run, because they reject NaN. If you only impute, the model cannot tell "TPM was 4" from "the API failed and we inserted the median." `constraint_observed` and `expression_observed` keep that bit. HistGradientBoosting can split on NaN natively; imputing it as well makes the three models see the same matrix, which keeps the comparison fair.

### 13. A variant is `2:166012108:C>G` in *SCN1A*, HGVS `c.3879+1G>C`. Walk through it.

*SCN1A* is on the minus strand. ClinVar's SPDI `NC_000002.12:166012107:C:G` is 0-based, so the 1-based GRCh38 base is 166012108, reference C, alternate G. On the coding strand that is G>C, and `c.3879+1` says it is the first base of the intron, the G of the GT. The canonical-dinucleotide flag is 1. The donor MaxEnt delta should be largely negative, because a G>C at +1 destroys the site. The benign-versus-pathogenic model does not need to be clever here. The interesting variants are the ones where this story is false.

### 14. How does this relate to AlphaGenome, and what did you deliberately not build?

AlphaGenome predicts splice usage and junction strength from a megabase of sequence, jointly with expression and chromatin, and it was evaluated on variant-effect benchmarks in the 2026 *Nature* paper. This project studies the same biological question on a rare-disease gene panel with features a person can draw on a whiteboard. It does not train a transformer, it does not download a megabase around every variant, and it does not present itself as a clinical caller. The interview point is knowing which tool answers which question.

### 15. How would you extend it next, without pretending the extension is done?

Three extensions are well scoped. First, indels: apply a deletion or insertion to the coding-strand window and rescore, with tests on a known *DMD* or *CFTR* splice indel. Second, transcript choice: score both the Ensembl canonical transcript and the MANE Select transcript and report when the distance disagrees. Third, a proper prospective split by submission date, so the model is trained on older ClinVar records and tested on newer ones. That is a harder generalization test than gene hold-out, and it needs the submission dates kept in the table. None of these are implemented. Saying they are not is part of the answer.

## 9. A short glossary

| Term | Meaning in this repo |
| --- | --- |
| Donor / 5′ splice site | Start of the intron. Canonical dinucleotide GT. |
| Acceptor / 3′ splice site | End of the intron. Canonical dinucleotide AG. |
| PWM | Independent-base information-content score. |
| MaxEntScan | Yeo & Burge maximum-entropy splice score. |
| PPT | Polypyrimidine tract, acceptor −20..−3. |
| YTRAC | The branchpoint pattern searched upstream of the acceptor. |
| SPDI | NCBI's sequence-position-deletion-insertion notation. Positions are 0-based. |
| GroupKFold | Cross-validation that keeps all rows of a gene together. |
| Brier score | Mean squared error between the label and the predicted probability. |
| Tau | Gene-level tissue specificity across GTEx tissues. |
| Deployment score | Full-cohort refit used by the app. Not the published metric. |

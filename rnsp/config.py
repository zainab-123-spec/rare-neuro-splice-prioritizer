"""Paths, gene panel, and API settings for the splice prioritizer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
REPORT_DIR = PROJECT_ROOT / "docs" / "reports"

MAXENT_DIR = PACKAGE_ROOT / "data" / "maxent"

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENSEMBL_REST = "https://rest.ensembl.org"
GTEX_API = "https://gtexportal.org/api/v2"
HPA_GENE_URL = "https://www.proteinatlas.org/{ensembl_id}.json"

TOOL_NAME = "rare-neuro-splice-prioritizer"
CONTACT_EMAIL = "276657223+zainab-123-spec@users.noreply.github.com"
USER_AGENT = f"{TOOL_NAME}/0.1 (educational regulatory-genomics project; mailto:{CONTACT_EMAIL})"

# NCBI asks unauthenticated clients to stay at or below 3 requests/second.
NCBI_MIN_INTERVAL_S = 0.5
ENSEMBL_MIN_INTERVAL_S = 0.2
GTEX_MIN_INTERVAL_S = 0.35
HPA_MIN_INTERVAL_S = 0.35

ENSEMBL_TILE_BP = 8_000
SEQUENCE_FLANK_BP = 120
JUNCTION_FLANK_BP = 70
CRYPTIC_FLANK_BP = 80
GC_FLANK_BP = 50

RANDOM_SEED = 42

CONSEQUENCES: tuple[str, ...] = (
    "splice donor variant",
    "splice acceptor variant",
    "splice region variant",
    "intron variant",
    "5 prime UTR variant",
    "3 prime UTR variant",
)

# Cap each gene × label × consequence slice. Documented in the README.
# Intron is the class that contains non-canonical +3..+50 and deep-intronic
# alleles (ClinVar does not index the phrase "splice region variant").
CONSEQUENCE_RETMAX: dict[str, int] = {
    "splice donor variant": 100,
    "splice acceptor variant": 100,
    "splice region variant": 60,
    "intron variant": 160,
    "5 prime UTR variant": 80,
    "3 prime UTR variant": 80,
}

POSITIVE_LABELS = frozenset({"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"})
NEGATIVE_LABELS = frozenset({"benign", "likely benign", "benign/likely benign"})

DONOR_OFFSETS: tuple[int, ...] = (-3, -2, -1, 1, 2, 3, 4, 5, 6)
ACCEPTOR_OFFSETS: tuple[int, ...] = tuple(range(-20, 0)) + (1, 2, 3)
PPT_OFFSETS: tuple[int, ...] = tuple(range(-20, -2))
# Branchpoint 5-mers start in this inclusive range relative to the acceptor.
BRANCHPOINT_STARTS: tuple[int, ...] = tuple(range(-48, -17))

# A new GT/AG whose MaxEntScan score clears this value raises a cryptic alert.
CRYPTIC_MAXENT_ALERT = 3.0

GTEX_BRAIN_TISSUES: tuple[str, ...] = (
    "Brain_Amygdala",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Caudate_basal_ganglia",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Cerebellum",
    "Brain_Cortex",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Hippocampus",
    "Brain_Hypothalamus",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Substantia_nigra",
)

GTEX_CORTEX = "Brain_Cortex"
GTEX_HIPPOCAMPUS = "Brain_Hippocampus"
GTEX_CEREBELLUM = "Brain_Cerebellum"


@dataclass(frozen=True)
class PanelGene:
    """One gene on the neurological / splice-disease panel."""

    symbol: str
    category: str
    disease: str
    aliases: tuple[str, ...] = ()


# Thirty genes requested for the panel. CFTR is the classic splice-disease
# benchmark on that list; it is not a brain gene, and the expression features
# are allowed to say so.
GENE_PANEL: tuple[PanelGene, ...] = (
    PanelGene("SCN1A", "epilepsy_neurodevelopmental", "Dravet syndrome / GEFS+"),
    PanelGene("DNM1", "epilepsy_neurodevelopmental", "Developmental and epileptic encephalopathy"),
    PanelGene("MECP2", "epilepsy_neurodevelopmental", "Rett syndrome"),
    PanelGene("STXBP1", "epilepsy_neurodevelopmental", "STXBP1 encephalopathy"),
    PanelGene("KCNQ2", "epilepsy_neurodevelopmental", "KCNQ2 neonatal encephalopathy"),
    PanelGene("SYNGAP1", "epilepsy_neurodevelopmental", "SYNGAP1-related disorder"),
    PanelGene("CDKL5", "epilepsy_neurodevelopmental", "CDKL5 deficiency disorder"),
    PanelGene("TSC1", "epilepsy_neurodevelopmental", "Tuberous sclerosis"),
    PanelGene("TSC2", "epilepsy_neurodevelopmental", "Tuberous sclerosis"),
    PanelGene("CHD2", "epilepsy_neurodevelopmental", "CHD2-related epilepsy"),
    PanelGene("SCN2A", "epilepsy_neurodevelopmental", "SCN2A-related disorder"),
    PanelGene("SCN8A", "epilepsy_neurodevelopmental", "SCN8A encephalopathy"),
    PanelGene("GABRG2", "epilepsy_neurodevelopmental", "GEFS+"),
    PanelGene("CACNA1A", "epilepsy_neurodevelopmental", "Episodic ataxia / CACNA1A encephalopathy"),
    PanelGene("DMD", "neuromuscular", "Duchenne muscular dystrophy"),
    PanelGene("SMN1", "neuromuscular", "Spinal muscular atrophy"),
    PanelGene("SOD1", "neurodegenerative", "Amyotrophic lateral sclerosis"),
    PanelGene("TARDBP", "neurodegenerative", "Amyotrophic lateral sclerosis"),
    PanelGene("FUS", "neurodegenerative", "Amyotrophic lateral sclerosis"),
    PanelGene("LRRK2", "neurodegenerative", "Parkinson disease"),
    PanelGene("PARK7", "neurodegenerative", "Parkinson disease"),
    PanelGene("PINK1", "neurodegenerative", "Parkinson disease"),
    PanelGene("HEXA", "neurometabolic", "Tay-Sachs disease"),
    PanelGene("PAH", "neurometabolic", "Phenylketonuria"),
    PanelGene(
        "GBA1", "neurometabolic", "Gaucher disease / GBA1-associated Parkinson disease", ("GBA",)
    ),
    PanelGene("ATP7B", "neurometabolic", "Wilson disease"),
    PanelGene("NPC1", "neurometabolic", "Niemann-Pick disease type C"),
    PanelGene("GALC", "neurometabolic", "Krabbe disease"),
    PanelGene("ABCD1", "neurometabolic", "X-linked adrenoleukodystrophy"),
    PanelGene("CFTR", "classic_splice_benchmark", "Cystic fibrosis"),
)

GENE_BY_SYMBOL = {gene.symbol: gene for gene in GENE_PANEL}

MODEL_FEATURES_FULL: tuple[str, ...] = (
    "abs_dist_to_splice",
    "nearest_is_donor",
    "disrupts_canonical_dinucleotide",
    "donor_maxent_delta",
    "acceptor_maxent_delta",
    "donor_pwm_delta",
    "acceptor_pwm_delta",
    "cryptic_donor_delta",
    "cryptic_acceptor_delta",
    "cryptic_donor_created",
    "cryptic_acceptor_created",
    "ppt_delta",
    "branchpoint_delta",
    "gc_delta",
    "cpg_delta",
    "gc_local",
    "motif_ref_information_bits",
    "constraint_overlap",
    "constraint_score",
    "constraint_observed",
    "gtex_brain_specificity",
    "gtex_brain_mean_tpm",
    "gtex_tau",
    "gtex_cortex_tpm",
    "gtex_hippocampus_tpm",
    "gtex_cerebellum_tpm",
    "hpa_brain_ntpm",
    "expression_observed",
)

GENE_LEVEL_FEATURES: frozenset[str] = frozenset(
    {
        "gtex_brain_specificity",
        "gtex_brain_mean_tpm",
        "gtex_tau",
        "gtex_cortex_tpm",
        "gtex_hippocampus_tpm",
        "gtex_cerebellum_tpm",
        "hpa_brain_ntpm",
        "expression_observed",
    }
)

MECHANISM_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in MODEL_FEATURES_FULL
    if feature not in GENE_LEVEL_FEATURES and feature != "abs_dist_to_splice"
)

MOTIF_FEATURES: tuple[str, ...] = tuple(
    feature
    for feature in MECHANISM_FEATURES
    if feature not in {"disrupts_canonical_dinucleotide", "nearest_is_donor"}
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full": MODEL_FEATURES_FULL,
    "mechanism": MECHANISM_FEATURES,
    "motif": MOTIF_FEATURES,
}

DISCLAIMER = (
    "Research and education tool only. This is not a medical device and it "
    "does not provide a diagnosis, a clinical classification, or a treatment "
    "recommendation. ClinVar labels are submitted assertions, not proof that "
    "a variant acts through splicing. Scores are cohort-relative probabilities "
    "from a small gene panel and must not be used to interpret a patient variant."
)

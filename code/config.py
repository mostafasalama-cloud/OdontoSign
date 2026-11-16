"""
Central configuration for the OdontoSign pipeline (DOLCHID).

Edit the PATHS section for your machine. Everything else is pre-specified
(fixed before evaluation) and is written to output/run_config.json so the
manuscript can report every setting exactly.

Environment variables (optional) override the paths, which is how the
synthetic smoke test runs:
    DOLCHID_DATA_ROOT, DOLCHID_HIST_DIR, DOLCHID_OUTPUT_DIR
"""
import os
from pathlib import Path

# =============================================================================
# PATHS  (edit these)
# =============================================================================
DATA_ROOT = Path(os.environ.get("DOLCHID_DATA_ROOT", "data/CBCT_All"))
CBCT_IMAGE_DIR = DATA_ROOT / "cbct_image"   # <ID>_CBCT_Image.nii.gz
CBCT_LABEL_DIR = DATA_ROOT / "cbct_label"   # <ID>_CBCT_Label.nii.gz (lesion masks)
HIST_DIR = Path(os.environ.get("DOLCHID_HIST_DIR", "data/hist_image"))  # <ID>_HIST_Image.png
OUTPUT_DIR = Path(os.environ.get("DOLCHID_OUTPUT_DIR", "pipeline_output"))

# Figshare file that contains the DOLCHID archive (used only by step0 to fetch
# the histopathology PNGs if you do not already have them).
FIGSHARE_ZIP_URL = "https://ndownloader.figshare.com/files/58072198"

# =============================================================================
# TASK / COHORT
# =============================================================================
# "AME_vs_DC" is the primary task of the paper. "ALL4" runs the four-class
# problem (AME, DC, KCOT, RC) as a secondary analysis requested by reviewers.
TASK = os.environ.get("DOLCHID_TASK", "AME_vs_DC")
TASK_CLASSES = {
    "AME_vs_DC": ["AME", "DC"],
    "ALL4": ["AME", "DC", "KCOT", "RC"],
}
# Primary cohort = every patient of the task classes that has CBCT image,
# CBCT lesion label and histopathology image (pre-specified inclusion rule).
# Set to True to reproduce the original manuscript subset (AME_1-72 and the
# 28 DC IDs used in the first submission) as a sensitivity analysis.
RESTRICT_TO_ORIGINAL_100 = os.environ.get("DOLCHID_ORIGINAL100", "0") == "1"
ORIGINAL_100_DC_IDS = [1, 2, 3] + list(range(10, 35))  # DC_4..DC_9 were absent

# =============================================================================
# PREPROCESSING  (all label-free; computed once per patient)
# =============================================================================
IMG_SIZE = 224                 # ResNet-18 input size
CBCT_N_SLICES = 9              # slices averaged into the 2D CBCT summary
CBCT_PERCENTILES = (1.0, 99.0) # per-scan intensity clipping
CBCT_PERCENTILE_SUBSAMPLE = 4  # voxel stride used to estimate percentiles
CBCT_SMOOTH_SIGMA = 1.0        # Gaussian smoothing of the reference before SLIC
CBCT_LESION_MARGIN = 0.25      # ROI margin (fraction of lesion bbox size)
CBCT_MIN_ROI = 32              # minimum ROI side (voxels)
# CBCT representations evaluated:
#   CBCT_U : uniform slices across the whole volume (original manuscript)
#   CBCT_L : slices spanning the annotated lesion + in-plane lesion ROI crop
CBCT_MODES = ["CBCT_U", "CBCT_L"]

HIST_SLIC_SIGMA = 3.0          # SLIC pre-smoothing for histopathology (suppresses nuclear-scale
                               # texture at 224 px; sigma=1 collapsed to ~5 regions)

# =============================================================================
# REGION PARTITIONING SCHEMES
# name -> dict(kind, K, compactness)
#   slic : SLIC superpixels (scikit-image), similarity + proximity
#   grid : fixed sqrt(K) x sqrt(K) square patches (content-agnostic)
#   rand : random Voronoi partition with K seeds (content-agnostic, irregular)
# scikit-image min-max rescales every input to [0,1] before SLIC. Histopathology
# is then converted to CIELAB (L* in 0-100), so compactness applies on that
# scale. For grayscale CBCT the intensity stays on [0,1]; compactness is
# therefore divided by 100 internally so the same nominal value means the same
# colour/space trade-off in both modalities.
# =============================================================================
SCHEMES = {
    "slic16":     dict(kind="slic", K=16, compactness=10.0),   # primary
    "grid16":     dict(kind="grid", K=16),
    "rand16":     dict(kind="rand", K=16),
    "slic8":      dict(kind="slic", K=8,  compactness=10.0),   # sensitivity
    "slic32":     dict(kind="slic", K=32, compactness=10.0),   # sensitivity
    "slic16_m5":  dict(kind="slic", K=16, compactness=5.0),    # sensitivity
    "slic16_m20": dict(kind="slic", K=16, compactness=20.0),   # sensitivity
}
# Region input construction:
#   masked : 224x224 image with pixels outside the region set to 0 (original)
#   crop   : bounding box of the region cropped from the native-resolution
#            image and resized to 224x224 (keeps cellular detail for histology)
REGION_INPUTS = {
    "slic16": ["masked", "crop"],
    "grid16": ["masked", "crop"],
    "rand16": ["masked"],
    "slic8": ["masked"],
    "slic32": ["masked"],
    "slic16_m5": ["masked"],
    "slic16_m20": ["masked"],
}
CONCAT_K = 16  # fixed count for the order-dependent concatenation baseline

# =============================================================================
# FEATURE EXTRACTOR (frozen)
# =============================================================================
BACKBONE = "resnet18"
PRETRAINED = os.environ.get("DOLCHID_NO_PRETRAINED", "0") != "1"  # ImageNet weights
FEATURE_DIM = 512
FEATURE_BATCH = 64
# "auto" picks xpu (Intel Arc) > cuda > cpu
FEATURE_DEVICE = os.environ.get("DOLCHID_FEATURE_DEVICE", "auto")

# =============================================================================
# ATTENTION-MIL MODEL (identical hyper-parameters to the original manuscript,
# fixed a priori; no tuning on the evaluation data)
# =============================================================================
MIL = dict(
    att_hidden=256,           # attention MLP: Linear(512,256)-Tanh-Linear(256,1)
    head=[256, 64],           # classifier: Linear(.,256)-ReLU-Drop-Linear(256,64)-ReLU-Drop-Linear(64,C)
    dropout=[0.3, 0.2],
    lr=1e-4,
    weight_decay=1e-5,
    epochs=100,               # fixed number of full-batch epochs, no early stopping
)
MIL_DEVICE = os.environ.get("DOLCHID_MIL_DEVICE", "cpu")  # tiny model: CPU is fastest

# =============================================================================
# LOGISTIC REGRESSION (for label-free representations)
# C chosen by inner 5-fold CV inside each outer training fold.
# =============================================================================
LR_CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LR_INNER_FOLDS = 5

# =============================================================================
# EVALUATION
# =============================================================================
SEED = 42
N_FOLDS = 5
N_REPEATS = int(os.environ.get("DOLCHID_N_REPEATS", "10"))  # repeated stratified 5-fold CV
N_JOBS = int(os.environ.get("DOLCHID_N_JOBS", "-1"))        # parallel workers (-1 = all cores)

# Statistics
N_PERMUTATIONS = int(os.environ.get("DOLCHID_N_PERM", "1000"))  # p-value floor = 1/(N+1)
N_BOOTSTRAP = int(os.environ.get("DOLCHID_N_BOOT", "2000"))
ALPHA = 0.05

# Representations given a full-pipeline permutation test (pre-specified).
PERMUTATION_REPS = [
    "HIST|slic16_masked|attn",
    "CBCT_U|slic16_masked|attn",
    "CBCT_L|slic16_masked|attn",
    "MM_L|slic16_masked|attn",
    "HIST|whole|lr",
    "CBCT_L|whole|lr",
    "MM_L|whole|lr",
]

# OdontoSign

Code and reader-study data for the paper:

> **OdontoSign: interpretable, box-prompted classification of odontogenic lesions from cone-beam CT using three-dimensional radiological-sign descriptors.**
> M. S. Abdelhady, M. Sharawi, N. Kotaem. *(submitted)*

OdontoSign classifies ameloblastoma (AME), odontogenic keratocyst (KCOT), dentigerous cyst (DC) and radicular cyst (RC) from preoperative CBCT:
a rough 3D box around the lesion prompts a 3D U-Net segmentation, 43 radiological-sign descriptors are computed from the segmented lesion,
and a regularised logistic regression returns class probabilities with a sign-level explanation.
All evaluations use leakage-free, 10-times-repeated stratified 5-fold cross-validation.

## Data

The DOLCHID dataset is **not** redistributed here. Download it from Figshare
([doi:10.6084/m9.figshare.30156622](https://doi.org/10.6084/m9.figshare.30156622)) and place it as:

```
data/CBCT_All/cbct_image/<ID>_CBCT_Image.nii.gz
data/CBCT_All/cbct_label/<ID>_CBCT_Label.nii.gz
data/hist_image/<ID>_HIST_Image.png
```

or point to other locations with the environment variables `DOLCHID_DATA_ROOT`, `DOLCHID_HIST_DIR` and `DOLCHID_OUTPUT_DIR` (see `code/config.py`).

## Installation

Python 3.12 and PyTorch (CPU, CUDA or Intel XPU build), then:

```
pip install -r code/requirements.txt
```

torchvision is not required. ImageNet ResNet-18 and DINOv2 (ViT-S/14, ViT-B/14) weights are downloaded automatically from their official URLs on first use.

## Running

Run from the `code` folder; every step is resumable and writes to `pipeline_output/`.
The `.bat` files are Windows double-click launchers for the same commands.

| Step | Script | What it does |
|---|---|---|
| 1–5 | `run_all.py --task ALL4` (and `run_all.py` for AME vs DC) | Cohort table; region-partitioned attention baselines (SLIC / grid / random regions); evaluation, statistics, report |
| 6 | `step6_cbct_features.py` | Radiological-sign descriptors, 2.5D views, 3D lesion cubes, histopathology ROI tiles |
| 7 | `step7_cbct_evaluate.py` | All CBCT, histopathology and fusion models on identical CV splits |
| 8 | `step8_cbct_3dcnn.py` | 3D CNN trained from scratch inside each fold |
| 9 | `step9_cbct_report.py` | Paired tests (corrected t-test + Holm), permutation tests, bootstrap CIs, benchmark table |
| 10 | `step10_cbct_extra.py` | Mask robustness, descriptor importance, univariate tests, calibration, decision curves |
| 11 | `step11_mask_augment.py` | Perturbation-augmented training and descriptor-family ablation |
| 12 | `step12_semiauto_seg.py` | Box-prompted 3D U-Net (out-of-fold) and OdontoSign on predicted masks |
| 13 | `step13_clinician_boxes.py` | Reader study: OdontoSign with clinician-drawn boxes |

`prepare_box_study.py` builds the anonymised, randomised scans given to the clinician.
`tests/make_synthetic_data.py` generates a small synthetic dataset for a quick smoke test.

## Reader study

`reader_study/` contains the 60 boxes drawn by the clinician in 3D Slicer (`boxes_clinician/case_XX_box.mrk.json`, LPS coordinates in mm),
the mapping from case number to DOLCHID patient ID (`case_key.csv`), the clinician's reading log,
the per-case box quality check and the instruction sheet given to the clinician.

## Licence

Code: MIT (see `LICENSE`). DOLCHID is subject to its own licence and terms of use.

## Citation

If you use this code, please cite the paper above and the DOLCHID dataset.

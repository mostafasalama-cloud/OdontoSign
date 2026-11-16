"""Create a tiny fake DOLCHID folder (NIfTI + PNG) for a smoke test."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

root = Path(sys.argv[1])
n = {"AME": 18, "DC": 10, "KCOT": 4, "RC": 4}
rng = np.random.default_rng(0)
for d in ["cbct_image", "cbct_label", "hist_image", "hist_label"]:
    (root / d).mkdir(parents=True, exist_ok=True)
for cls, k in n.items():
    for i in range(1, k + 1):
        pid = f"{cls}_{i}"
        sig = 1.0 if cls == "DC" else 0.0
        vol = rng.normal(500, 80, (48, 40, 30)).astype(np.float32)
        lab = np.zeros(vol.shape, np.uint8)
        lab[15:28, 12:24, 8:20] = 1
        vol[lab > 0] += 400 + 300 * sig  # lesion brightness differs by class
        aff = np.diag([-0.3, 0.3, 0.3, 1.0])   # LPS-like, tests reorientation
        nib.save(nib.Nifti1Image(vol, aff), root / "cbct_image" / f"{pid}_CBCT_Image.nii.gz")
        nib.save(nib.Nifti1Image(lab, aff), root / "cbct_label" / f"{pid}_CBCT_Label.nii.gz")
        if cls == "DC" and i == 10:
            continue  # one missing histology -> exclusion path
        img = rng.integers(150, 230, (600, 800, 3)).astype(np.uint8)
        img[200:400, 200:500, 0] = 120 + int(80 * sig) + int(30 * (cls == "KCOT"))
        Image.fromarray(img).save(root / "hist_image" / f"{pid}_HIST_Image.png")
        roi = np.zeros(img.shape[:2], np.uint8); roi[40:280, 60:380] = 255
        Image.fromarray(roi).save(root / "hist_label" / f"{pid}_HIST_Label.png")
print("synthetic data in", root)

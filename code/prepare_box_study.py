"""
Prepare the clinician box-drawing study (reader study for OdontoSign).

Selects 60 cases (15 per class, fixed seed), removes every hint of the diagnosis,
downsamples the CBCT volumes 2x (0.6 mm, about 20 MB each) so the package is easy to
share, and writes them as case_01.nii.gz ... case_60.nii.gz in random order.

Outputs (<project>/box_study):
  for_clinician\\case_XX.nii.gz      volumes to send (no masks, no labels)
  for_clinician\\reading_log.csv     sheet the clinician fills in
  PRIVATE_key.csv                    case -> patient ID and class (do NOT send)
The physical coordinates (affine) are preserved, so boxes drawn on the 0.6 mm volumes
map exactly onto the original 0.3 mm volumes.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import csv
import numpy as np
import nibabel as nib

import config as C

N_PER_CLASS = 15
SEED = 2026
CLASSES = ["AME", "DC", "KCOT", "RC"]
OUT = _os.path.join(_os.path.dirname(str(C.OUTPUT_DIR)), "box_study")
IMG_DIR = _os.path.join(str(C.DATA_ROOT), "cbct_image")


def downsample2(vol):
    s = [d - d % 2 for d in vol.shape]
    v = vol[: s[0], : s[1], : s[2]].astype(np.float32)
    v = v.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5))
    return v


def main():
    rng = np.random.default_rng(SEED)
    files = sorted(f for f in _os.listdir(IMG_DIR) if f.endswith("_CBCT_Image.nii.gz"))
    pids = [f.replace("_CBCT_Image.nii.gz", "") for f in files]
    chosen = []
    for c in CLASSES:
        pool = sorted(p for p in pids if p.split("_")[0] == c)
        chosen += list(rng.choice(pool, N_PER_CLASS, replace=False))
    rng.shuffle(chosen)

    fc = _os.path.join(OUT, "for_clinician")
    _os.makedirs(fc, exist_ok=True)
    key = []
    for i, pid in enumerate(chosen, 1):
        case = f"case_{i:02d}"
        dst = _os.path.join(fc, f"{case}.nii.gz")
        key.append(dict(case=case, patient_id=pid, clinical_class=pid.split("_")[0]))
        if _os.path.exists(dst):
            print(f"[{i:2d}/{len(chosen)}] {case} exists, skipped", flush=True)
            continue
        img = nib.load(_os.path.join(IMG_DIR, f"{pid}_CBCT_Image.nii.gz"))
        vol = np.asanyarray(img.dataobj)
        dtype = vol.dtype
        small = downsample2(vol)
        if np.issubdtype(dtype, np.integer):
            small = np.round(small).astype(dtype)
        aff = img.affine.copy()
        new_aff = aff @ np.diag([2.0, 2.0, 2.0, 1.0])
        new_aff[:3, 3] = (aff @ np.array([0.5, 0.5, 0.5, 1.0]))[:3]
        out = nib.Nifti1Image(small, new_aff)
        out.header.set_xyzt_units("mm")
        out.header["descrip"] = b""
        nib.save(out, dst)
        print(f"[{i:2d}/{len(chosen)}] {case}  {small.shape}  {_os.path.getsize(dst)/1e6:.0f} MB", flush=True)

    with open(_os.path.join(OUT, "PRIVATE_key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "patient_id", "clinical_class"])
        w.writeheader(); w.writerows(key)
    with open(_os.path.join(fc, "reading_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "lesion_found (yes/no)", "seconds_to_draw (approx.)",
                    "confidence_box_covers_lesion (1=low 2=medium 3=high)", "comments"])
        for k in key:
            w.writerow([k["case"], "", "", "", ""])
    print(f"Done. Send the folder {fc}. Keep PRIVATE_key.csv to yourself.", flush=True)


if __name__ == "__main__":
    main()

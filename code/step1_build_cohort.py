"""
STEP 1: Build the patient index and the case-selection flow.

Matches CBCT image, CBCT lesion label and histopathology image by patient ID
(e.g. AME_12) and records image metadata (shape, voxel spacing, orientation,
histology resolution). Header-only reads, so this step is fast.

Outputs (output/cohort/):
    cohort_all.csv        one row per patient ID found in any folder
    cohort_flow.csv       per-class counts at each inclusion step
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import nibabel as nib
import pandas as pd
from PIL import Image

import config as C
from common import environment_info, log, out_dir, parse_id, save_json

Image.MAX_IMAGE_PIXELS = None


def scan(folder, kind):
    found = {}
    if not folder.exists():
        log(f"WARNING: folder not found: {folder}")
        return found
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        parsed = parse_id(p.name)
        if parsed is None:
            continue
        pid = parsed[0]
        if pid in found:
            log(f"WARNING: duplicate {kind} file for {pid}: {p.name}")
            continue
        found[pid] = p
    return found


def main():
    log("Scanning folders")
    imgs = scan(C.CBCT_IMAGE_DIR, "CBCT image")
    lbls = scan(C.CBCT_LABEL_DIR, "CBCT label")
    hists = {pid: p for pid, p in scan(C.HIST_DIR, "histology").items()
             if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")}
    log(f"CBCT images: {len(imgs)} | CBCT labels: {len(lbls)} | histology: {len(hists)}")
    if not hists:
        log("No histopathology images found. Run step0_download_histology.py or copy the "
            f"PNGs into {C.HIST_DIR}")

    rows = []
    for pid in sorted(set(imgs) | set(lbls) | set(hists),
                      key=lambda s: (s.split("_")[0], int(s.split("_")[1]))):
        cls, num = pid.split("_")[0], int(pid.split("_")[1])
        r = dict(patient_id=pid, clinical_class=cls, case_number=num,
                 cbct_path=str(imgs.get(pid, "")), label_path=str(lbls.get(pid, "")),
                 hist_path=str(hists.get(pid, "")),
                 has_cbct=pid in imgs, has_label=pid in lbls, has_hist=pid in hists,
                 header_ok=True, exclusion_reason="")
        try:
            if pid in imgs:
                im = nib.load(imgs[pid])
                r["cbct_shape"] = "x".join(map(str, im.shape))
                r["cbct_spacing_mm"] = "x".join(f"{z:.3f}" for z in im.header.get_zooms()[:3])
                r["cbct_orientation"] = "".join(nib.aff2axcodes(im.affine))
                if pid in lbls:
                    lb = nib.load(lbls[pid])
                    if lb.shape[:3] != im.shape[:3]:
                        r["header_ok"] = False
                        r["exclusion_reason"] = "label shape differs from image"
            if pid in hists:
                with Image.open(hists[pid]) as h:
                    r["hist_width"], r["hist_height"] = h.size
                    r["hist_mode"] = h.mode
        except Exception as e:
            r["header_ok"] = False
            r["exclusion_reason"] = f"unreadable file: {e}"
        if not r["exclusion_reason"]:
            missing = [n for n, ok in [("CBCT", r["has_cbct"]), ("label", r["has_label"]),
                                        ("histology", r["has_hist"])] if not ok]
            if missing:
                r["exclusion_reason"] = "missing " + ", ".join(missing)
        r["eligible"] = r["exclusion_reason"] == ""
        rows.append(r)

    df = pd.DataFrame(rows)
    out = out_dir("cohort")
    df.to_csv(out / "cohort_all.csv", index=False)

    flow = []
    for cls, g in df.groupby("clinical_class"):
        flow.append(dict(
            clinical_class=cls,
            ids_found=len(g),
            with_cbct=int(g.has_cbct.sum()),
            with_cbct_and_label=int((g.has_cbct & g.has_label).sum()),
            with_cbct_label_hist=int((g.has_cbct & g.has_label & g.has_hist).sum()),
            readable_and_consistent=int(g.eligible.sum()),
        ))
    flow = pd.DataFrame(flow)
    flow.loc[len(flow)] = ["TOTAL"] + flow.drop(columns="clinical_class").sum().tolist()
    flow.to_csv(out / "cohort_flow.csv", index=False)
    save_json(environment_info(), out / "environment.json")

    print("\nCase-selection flow:")
    print(flow.to_string(index=False))
    excl = df[~df.eligible]
    if len(excl):
        print(f"\nExcluded IDs ({len(excl)}):")
        print(excl[["patient_id", "exclusion_reason"]].to_string(index=False))
    log(f"Saved {out / 'cohort_all.csv'}")


if __name__ == "__main__":
    main()

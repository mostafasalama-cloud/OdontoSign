"""
STEP 2: Label-free preprocessing, region partitioning and frozen feature extraction.

Nothing in this step uses class labels, and every operation is per patient
(per-scan intensity percentiles, per-image SLIC, frozen ImageNet ResNet-18).
It is therefore computed once for all patients; all label-dependent learning
happens later, inside the cross-validation training folds (step 3).

For every patient it stores, per modality (CBCT_U, CBCT_L, HIST):
    whole                 (1, 512)  whole-image feature (direct baseline)
    <scheme>_<input>      (K, 512)  one feature vector per region
    seg_<scheme>          (224,224) region label map (for figures/audit)
    preview               224x224 image used for partitioning (for figures)
    meta                  JSON with slice indices, ROI, percentiles, region counts

Resumable: patients already processed with the same settings are skipped.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import argparse
import hashlib
import json
import zlib

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import slic
from skimage.transform import resize

import config as C
from common import feature_path, log, out_dir, pick_device, set_seed

Image.MAX_IMAGE_PIXELS = None
S = C.IMG_SIZE
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def settings_hash():
    keys = dict(IMG_SIZE=C.IMG_SIZE, N=C.CBCT_N_SLICES, P=C.CBCT_PERCENTILES,
                SUB=C.CBCT_PERCENTILE_SUBSAMPLE, SIG=C.CBCT_SMOOTH_SIGMA,
                MARGIN=C.CBCT_LESION_MARGIN, MINROI=C.CBCT_MIN_ROI, HS=C.HIST_SLIC_SIGMA,
                SCHEMES=C.SCHEMES, INPUTS=C.REGION_INPUTS, BB=C.BACKBONE, PRE=C.PRETRAINED)
    return hashlib.md5(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:10]


# =============================================================================
# Feature extractor
# =============================================================================
class Extractor:
    def __init__(self):
        import torch
        from resnet import resnet18
        self.torch = torch
        self.device = pick_device(C.FEATURE_DEVICE)
        m = resnet18(pretrained=C.PRETRAINED)  # torchvision-identical, ImageNet1K_V1
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        log(f"Feature extractor: ResNet-18 ({'ImageNet' if C.PRETRAINED else 'RANDOM'} weights, "
            f"frozen) on {self.device}")

    def __call__(self, images):
        """images: list of HxWx3 float arrays in [0,1] (already 224x224)."""
        torch = self.torch
        x = (np.stack(images).astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        feats = []
        with torch.inference_mode():
            for i in range(0, len(x), C.FEATURE_BATCH):
                feats.append(self.model(x[i:i + C.FEATURE_BATCH].to(self.device)).float().cpu())
        return torch.cat(feats).numpy().astype(np.float32)


# =============================================================================
# Image helpers
# =============================================================================
def to_rgb(img):
    return np.repeat(img[..., None], 3, axis=2) if img.ndim == 2 else img


def resize_float(img, shape):
    if img.ndim == 2:
        return resize(img, shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    return np.asarray(pil.resize((shape[1], shape[0]), Image.LANCZOS), dtype=np.float32) / 255.0


def partition(work, modality, pid, scheme):
    """Return a (S,S) int label map with consecutive labels starting at 0."""
    spec = C.SCHEMES[scheme]
    K = spec["K"]
    if spec["kind"] == "slic":
        if modality == "HIST":
            # RGB in [0,1]; scikit-image converts RGB -> CIELAB internally.
            seg = slic(work, n_segments=K, compactness=spec["compactness"],
                       sigma=C.HIST_SLIC_SIGMA, channel_axis=-1, convert2lab=True,
                       start_label=0, enforce_connectivity=True)
        else:
            # Grayscale CBCT: skimage rescales to [0,1], so compactness / 100
            # matches the CIELAB L* (0-100) scale used for histopathology.
            seg = slic(work, n_segments=K, compactness=spec["compactness"] / 100.0,
                       sigma=C.CBCT_SMOOTH_SIGMA, channel_axis=None, start_label=0,
                       enforce_connectivity=True)
    elif spec["kind"] == "grid":
        g = int(round(np.sqrt(K)))
        idx = np.minimum((np.arange(S) * g) // S, g - 1)
        seg = idx[:, None] * g + idx[None, :]
    elif spec["kind"] == "rand":
        rng = np.random.default_rng(zlib.crc32(f"{pid}|{modality}|{scheme}".encode()))
        pts = rng.uniform(0, S, size=(K, 2))
        yy, xx = np.mgrid[0:S, 0:S]
        d = (yy[..., None] - pts[:, 0]) ** 2 + (xx[..., None] - pts[:, 1]) ** 2
        seg = d.argmin(-1)
    else:
        raise ValueError(spec["kind"])
    _, seg = np.unique(seg, return_inverse=True)
    return seg.reshape(S, S).astype(np.int16)


def region_inputs(native, work, seg, mode):
    """Build one 224x224x3 image per region."""
    out = []
    H, W = native.shape[:2]
    for r in range(int(seg.max()) + 1):
        m = seg == r
        if mode == "masked":
            img = to_rgb(work) * m[..., None]
        else:  # crop from native resolution
            rows, cols = np.where(m)
            r0 = int(np.floor(rows.min() * H / S)); r1 = int(np.ceil((rows.max() + 1) * H / S))
            c0 = int(np.floor(cols.min() * W / S)); c1 = int(np.ceil((cols.max() + 1) * W / S))
            r1, c1 = max(r1, r0 + 2), max(c1, c0 + 2)
            img = to_rgb(resize_float(native[r0:r1, c0:c1], (S, S)))
        out.append(img.astype(np.float32))
    return out


# =============================================================================
# Modality loaders  ->  (native image, meta)
# =============================================================================
def load_cbct_pair(row):
    im = nib.load(row.cbct_path)
    vol = im.get_fdata(dtype=np.float32)
    ornt = nib.orientations.io_orientation(im.affine)          # -> RAS+
    vol = nib.orientations.apply_orientation(vol, ornt)
    zooms_in = np.array(im.header.get_zooms()[:3], dtype=float)
    zooms = np.empty(3); zooms[ornt[:, 0].astype(int)] = zooms_in
    lb = nib.load(row.label_path)
    lab = np.asanyarray(lb.dataobj)
    lab = nib.orientations.apply_orientation(lab, nib.orientations.io_orientation(lb.affine)) > 0

    s = C.CBCT_PERCENTILE_SUBSAMPLE
    lo, hi = np.percentile(vol[::s, ::s, ::s], C.CBCT_PERCENTILES)
    D = vol.shape[2]  # axial axis after RAS reorientation
    norm = lambda a: np.clip((a - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

    base_meta = dict(shape_ras="x".join(map(str, vol.shape)),
                     spacing_ras_mm="x".join(f"{z:.3f}" for z in zooms),
                     orientation_in="".join(nib.aff2axcodes(im.affine)),
                     p_low=float(lo), p_high=float(hi))
    outputs = {}

    # --- CBCT_U: uniform slices over the whole volume (original manuscript) ---
    zs = np.linspace(0, D - 1, C.CBCT_N_SLICES).astype(int)
    outputs["CBCT_U"] = (np.mean([norm(vol[:, :, z]) for z in zs], axis=0),
                         dict(base_meta, slice_indices=zs.tolist(), roi="full"))

    # --- CBCT_L: slices spanning the annotated lesion + in-plane ROI ---
    zl = np.where(lab.any(axis=(0, 1)))[0]
    if len(zl) == 0:
        outputs["CBCT_L"] = (outputs["CBCT_U"][0],
                             dict(outputs["CBCT_U"][1], lesion_found=False))
    else:
        z0, z1 = int(zl.min()), int(zl.max())
        zs = np.linspace(z0, z1, C.CBCT_N_SLICES).round().astype(int)
        proj = lab[:, :, z0:z1 + 1].any(axis=2)
        rr, cc = np.where(proj)
        side = max(rr.max() - rr.min() + 1, cc.max() - cc.min() + 1)
        side = int(max(C.CBCT_MIN_ROI, side * (1 + 2 * C.CBCT_LESION_MARGIN)))
        side = min(side, vol.shape[0], vol.shape[1])
        cr, ccen = (rr.min() + rr.max()) // 2, (cc.min() + cc.max()) // 2
        r0 = int(np.clip(cr - side // 2, 0, vol.shape[0] - side))
        c0 = int(np.clip(ccen - side // 2, 0, vol.shape[1] - side))
        roi = np.mean([norm(vol[r0:r0 + side, c0:c0 + side, z]) for z in zs], axis=0)
        outputs["CBCT_L"] = (roi, dict(base_meta, slice_indices=zs.tolist(),
                                       lesion_z_range=[z0, z1],
                                       roi=[r0, r0 + side, c0, c0 + side],
                                       lesion_voxels=int(lab.sum()), lesion_found=True))
    del vol, lab
    return outputs


def load_hist(row):
    with Image.open(row.hist_path) as h:
        rgb = np.asarray(h.convert("RGB"), dtype=np.float32) / 255.0
    return rgb, dict(original_size=f"{rgb.shape[1]}x{rgb.shape[0]}")


# =============================================================================
# Per-modality processing
# =============================================================================
def process(native, meta, modality, pid, extractor):
    work = resize_float(native, (S, S))
    images, index = [to_rgb(work)], [("whole", 0, 1)]
    store, n_regions = {}, {}
    for scheme in C.SCHEMES:
        seg = partition(work, modality, pid, scheme)
        store[f"seg_{scheme}"] = seg
        n_regions[scheme] = int(seg.max()) + 1
        for mode in C.REGION_INPUTS[scheme]:
            imgs = region_inputs(native, work, seg, mode)
            index.append((f"{scheme}_{mode}", len(images), len(imgs)))
            images.extend(imgs)
    feats = extractor(images)
    for key, start, n in index:
        store[key] = feats[start:start + n].astype(np.float16)
    store["preview"] = (np.clip(work, 0, 1) * 255).astype(np.uint8)
    meta = dict(meta, n_regions=n_regions, settings=settings_hash())
    store["meta"] = np.array(json.dumps(meta))
    return store


def already_done(modality, pid):
    p = feature_path(modality, pid)
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return json.loads(str(z["meta"]))["settings"] == settings_hash()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-only", action="store_true",
                    help="extract only the classes of config.TASK (default: all eligible)")
    ap.add_argument("--limit", type=int, default=0, help="process only N patients (testing)")
    args = ap.parse_args()

    set_seed(C.SEED)
    df = pd.read_csv(out_dir("cohort") / "cohort_all.csv")
    df = df[df.eligible]
    if args.task_only:
        df = df[df.clinical_class.isin(C.TASK_CLASSES[C.TASK])]
    if args.limit:
        df = df.head(args.limit)
    log(f"Patients to process: {len(df)}  (settings {settings_hash()})")

    extractor = Extractor()
    zero = extractor([np.zeros((S, S, 3), np.float32)])
    for mod in C.CBCT_MODES + ["HIST"]:
        np.savez(out_dir("features", mod) / "_zero_image_feature.npz", feature=zero[0])

    for i, row in enumerate(df.itertuples(), 1):
        pid = row.patient_id
        todo_cbct = [m for m in C.CBCT_MODES if not already_done(m, pid)]
        todo_hist = not already_done("HIST", pid)
        if not todo_cbct and not todo_hist:
            continue
        log(f"[{i}/{len(df)}] {pid}")
        try:
            if todo_cbct:
                outs = load_cbct_pair(row)
                for mod in todo_cbct:
                    native, meta = outs[mod]
                    np.savez_compressed(feature_path(mod, pid),
                                        **process(native, meta, mod, pid, extractor))
            if todo_hist:
                native, meta = load_hist(row)
                np.savez_compressed(feature_path("HIST", pid),
                                    **process(native, meta, "HIST", pid, extractor))
        except MemoryError:
            log(f"  MemoryError on {pid}; close other programs and rerun (resumable).")
            raise

    # metadata table (answers the reviewer's reproducibility questions)
    rows = []
    for row in df.itertuples():
        for mod in C.CBCT_MODES + ["HIST"]:
            p = feature_path(mod, row.patient_id)
            if p.exists():
                with np.load(p) as z:
                    m = json.loads(str(z["meta"]))
                nr = m.pop("n_regions")
                rows.append(dict(patient_id=row.patient_id, modality=mod,
                                 **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                    for k, v in m.items()},
                                 **{f"n_regions_{k}": v for k, v in nr.items()}))
    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(out_dir("cohort") / "preprocessing_metadata.csv", index=False)
    summ = meta_df.groupby("modality")[[c for c in meta_df if c.startswith("n_regions_")]]\
        .agg(["mean", "min", "max"]).round(2)
    summ.to_csv(out_dir("cohort") / "region_count_summary.csv")
    print("\nRegions actually produced per scheme (mean/min/max):")
    print(summ.T.to_string())
    log("Step 2 complete.")


if __name__ == "__main__":
    main()

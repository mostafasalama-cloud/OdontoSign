"""
STEP 6 (CBCT-centred study): label-free lesion-level representations.

Per patient (all computed without class labels, per patient only):

  RAD      3D lesion descriptors from the CBCT volume + expert lesion mask
           (shape, size, intensity, rim, boundary sharpness, hyperdense
           content such as included teeth, GLCM texture, location)
  C25_*    2.5D multi-plane lesion views: 5 slice positions across the lesion
           in each of the axial / coronal / sagittal planes (15 views); each
           view = 3 adjacent slices as RGB channels, square crop around the
           lesion with margin, encoded by a frozen backbone
           (r18 = ResNet-18 ImageNet, dinos = DINOv2 ViT-S/14, dinob = DINOv2 ViT-B/14)
  HT_*     histopathology tiles (224 px, native resolution) inside the
           pathologists' ROI mask (up to 64 tiles) + one downsampled ROI view,
           same backbones (upper-bound reference only)
  CBCT3D   lesion-centred cube (64^3, isotropic, image + mask) for the 3D CNN

Resumable; output in pipeline_output/features/<SRC>/<ID>.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth)

import argparse
import hashlib
import json

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes, mesh_surface_area

import config as C
from common import log, out_dir, pick_device, set_seed

Image.MAX_IMAGE_PIXELS = None
S = 224
BACKBONES = ["r18", "dinos", "dinob"]
N_POS = 5            # slice positions per plane
ADJ = 2              # voxel offset of the adjacent slices used as channels
MARGIN = 0.25        # crop margin (fraction of lesion extent per side)
MIN_SIDE_MM = 15.0
CUBE = 64
MAX_TILES = 64
HIST_LABEL_DIR = C.HIST_DIR.parent / "hist_label"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
VERSION = hashlib.md5(json.dumps(dict(N_POS=N_POS, ADJ=ADJ, MARGIN=MARGIN, MIN=MIN_SIDE_MM, CUBE=CUBE,
                                      T=MAX_TILES, B=BACKBONES, v=3)).encode()).hexdigest()[:8]


# =============================================================================
# Encoders
# =============================================================================
class Encoders:
    def __init__(self, pretrained=True):
        import torch
        from dinov2_vit import dinov2
        from resnet import resnet18
        self.torch = torch
        self.dev = pick_device(C.FEATURE_DEVICE)
        r18 = resnet18(pretrained=pretrained)
        r18.fc = torch.nn.Identity()
        self.models = {"r18": r18, "dinos": dinov2("dinov2_vits14", pretrained),
                       "dinob": dinov2("dinov2_vitb14", pretrained)}
        for m in self.models.values():
            m.eval().to(self.dev)
            for p in m.parameters():
                p.requires_grad_(False)
        log(f"Encoders on {self.dev}: " + ", ".join(self.models))

    def __call__(self, images):
        torch = self.torch
        x = (np.stack(images).astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        out = {}
        with torch.inference_mode():
            for name, m in self.models.items():
                f = [m(x[i:i + 32].to(self.dev)).float().cpu() for i in range(0, len(x), 32)]
                out[name] = torch.cat(f).numpy().astype(np.float16)
        return out


def rs(img, size=(S, S)):
    """Resize a float [0,1] HxW or HxWx3 image."""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, 2)
    pil = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    return np.asarray(pil.resize(size, Image.BILINEAR), np.float32) / 255.0


# =============================================================================
# CBCT
# =============================================================================
def load_cbct(row):
    im = nib.load(row.cbct_path)
    ornt = nib.orientations.io_orientation(im.affine)
    vol = nib.orientations.apply_orientation(im.get_fdata(dtype=np.float32), ornt)
    zin = np.array(im.header.get_zooms()[:3], float)
    sp = np.empty(3); sp[ornt[:, 0].astype(int)] = zin
    lb = nib.load(row.label_path)
    lab = nib.orientations.apply_orientation(np.asanyarray(lb.dataobj),
                                             nib.orientations.io_orientation(lb.affine)) > 0
    lo, hi = np.percentile(vol[::4, ::4, ::4], [1, 99])
    p995 = np.percentile(vol[::4, ::4, ::4], 99.5)
    return vol, lab, sp, lo, hi, p995


def keep_largest(lab):
    cc, n = ndi.label(lab)
    if n <= 1:
        return lab, n
    sizes = ndi.sum(lab, cc, range(1, n + 1))
    return cc == (np.argmax(sizes) + 1), n


def glcm_feats(img2d, m2d, levels=32):
    vals = img2d[m2d]
    if m2d.sum() < 30:
        return None
    lo, hi = np.percentile(vals, [1, 99])
    q = np.clip((img2d - lo) / (hi - lo + 1e-6), 0, 1)
    q = (q * (levels - 2)).astype(np.uint8) + 1
    q[~m2d] = 0
    g = graycomatrix(q, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=levels,
                     symmetric=True, normed=False).astype(float)
    g = g[1:, 1:]  # drop pairs involving background
    g /= np.maximum(g.sum(axis=(0, 1), keepdims=True), 1)
    return {p: float(graycoprops(g, p).mean()) for p in
            ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")}


def radiomics(vol, lab, sp, lo, hi, p995):
    f = {}
    lab, n_comp = keep_largest(lab)
    f["n_mask_components"] = n_comp
    idx = np.argwhere(lab)
    vvol = float(np.prod(sp))
    cen = idx.mean(0)
    f.update({f"loc_{a}": cen[i] / lab.shape[i] for i, a in enumerate("xyz")})
    f["loc_abs_x"] = abs(cen[0] / lab.shape[0] - 0.5)
    # sub-volume around the lesion (+10 mm)
    pad = np.ceil(10 / sp).astype(int)
    b0 = np.maximum(idx.min(0) - pad, 0); b1 = np.minimum(idx.max(0) + pad + 1, lab.shape)
    sl = tuple(slice(a, b) for a, b in zip(b0, b1))
    v = (vol[sl] - lo) / (hi - lo + 1e-6)
    m = lab[sl]
    # --- shape
    V = m.sum() * vvol
    f["volume_mm3"] = V; f["log_volume"] = np.log(V + 1)
    try:
        verts, faces, _, _ = marching_cubes(np.pad(m, 1).astype(np.uint8), 0.5, spacing=tuple(sp))
        A = mesh_surface_area(verts, faces)
        f["surface_mm2"] = A
        f["sphericity"] = (np.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / A
        tri = verts[faces]  # mesh volume (divergence theorem) for a like-for-like ratio
        vmesh = abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
        f["solidity"] = vmesh / ConvexHull(verts).volume
    except Exception:
        f["surface_mm2"] = f["sphericity"] = f["solidity"] = np.nan
    pts = (idx - cen) * sp
    ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))[::-1]
    ax = 4 * np.sqrt(np.maximum(ev, 1e-9))
    f["axis_major_mm"], f["axis_minor_mm"], f["axis_least_mm"] = ax
    f["elongation"] = ax[1] / ax[0]; f["flatness"] = ax[2] / ax[0]
    ext = (idx.max(0) - idx.min(0) + 1) * sp
    f["extent"] = V / np.prod(ext)
    # --- intensity inside lesion
    iv = v[m]
    f.update(int_mean=iv.mean(), int_sd=iv.std(), int_skew=skew(iv), int_kurt=kurtosis(iv),
             **{f"int_p{q}": np.percentile(iv, q) for q in (10, 25, 50, 75, 90)})
    h, _ = np.histogram(iv, bins=32, range=(0, 1)); h = h / h.sum()
    f["int_entropy"] = float(-(h[h > 0] * np.log2(h[h > 0])).sum())
    # --- rim / surroundings
    r1 = np.round(1.5 / sp).astype(int).max()
    d1 = ndi.binary_dilation(m, iterations=r1)
    d2 = ndi.binary_dilation(d1, iterations=2 * r1)
    rim, outer = d1 & ~m, d2 & ~d1
    f["rim_mean"] = v[rim].mean(); f["rim_sd"] = v[rim].std()
    f["outer_mean"] = v[outer].mean()
    f["rim_contrast"] = f["rim_mean"] - f["int_mean"]
    f["frac_inside_above_rim"] = float((iv > f["rim_mean"]).mean())   # septa / bony content
    gm = ndi.gaussian_gradient_magnitude(v, 1.0)
    surf = m & ~ndi.binary_erosion(m)
    f["boundary_grad_mean"] = gm[surf].mean(); f["boundary_grad_sd"] = gm[surf].std()
    # --- hyperdense content (e.g. tooth crown in dentigerous cysts)
    hd = vol[sl] > p995
    f["hyperdense_frac_in"] = float(hd[m].mean())
    f["hyperdense_frac_rim"] = float(hd[d1].mean())
    f["hyperdense_mm3_rim"] = float(hd[d1].sum() * vvol)
    cc, ncc = ndi.label(hd & d1)
    f["hyperdense_n_objects"] = ncc
    # --- texture (GLCM on the central slice of each plane + 3 extra axial)
    tex = []
    c = np.round(np.argwhere(m).mean(0)).astype(int)
    for axis in range(3):
        for off in ([0] if axis < 2 else [-2, -1, 0, 1, 2]):
            ext_ax = np.where(m.any(axis=tuple(a for a in range(3) if a != axis)))[0]
            pos = int(np.clip(c[axis] + off * max(1, len(ext_ax) // 6), ext_ax.min(), ext_ax.max()))
            g = glcm_feats(np.take(v, pos, axis), np.take(m, pos, axis))
            if g:
                tex.append(g)
    for k in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        f[f"glcm_{k}"] = float(np.mean([t[k] for t in tex])) if tex else np.nan
    return {k: float(val) for k, val in f.items()}, lab


def views_25d(vol, lab, sp, lo, hi):
    idx = np.argwhere(lab)
    lo_i, hi_i = idx.min(0), idx.max(0)
    side_vox = np.ceil(max((hi_i - lo_i + 1).max() * (1 + 2 * MARGIN), MIN_SIDE_MM / sp.min())).astype(int)
    cen = ((lo_i + hi_i) / 2).astype(int)
    norm = lambda a: np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
    views, meta = [], []
    for axis in range(3):
        positions = np.linspace(lo_i[axis] + 0.1 * (hi_i[axis] - lo_i[axis]),
                                lo_i[axis] + 0.9 * (hi_i[axis] - lo_i[axis]), N_POS).round().astype(int)
        others = [a for a in range(3) if a != axis]
        box = []
        for a in others:
            s0 = int(np.clip(cen[a] - side_vox // 2, 0, max(vol.shape[a] - side_vox, 0)))
            box.append(slice(s0, s0 + side_vox))
        for p in positions:
            chans = []
            for d in (-ADJ, 0, ADJ):
                q = int(np.clip(p + d, 0, vol.shape[axis] - 1))
                sl = [None] * 3; sl[axis] = q; sl[others[0]] = box[0]; sl[others[1]] = box[1]
                chans.append(norm(vol[tuple(sl)]))
            img = np.stack(chans, -1)
            views.append(rs(img)); meta.append((axis, int(p)))
    return views, dict(side_vox=int(side_vox), views=meta)


def cube3d(vol, lab, sp, lo, hi):
    idx = np.argwhere(lab)
    lo_i, hi_i = idx.min(0), idx.max(0)
    side = int(np.ceil(max((hi_i - lo_i + 1).max() * (1 + 2 * MARGIN), MIN_SIDE_MM / sp.min())))
    cen = ((lo_i + hi_i) / 2).astype(int)
    pad = side
    vp = np.pad(vol, pad, mode="constant", constant_values=lo)
    lp = np.pad(lab, pad)
    s = tuple(slice(c + pad - side // 2, c + pad - side // 2 + side) for c in cen)
    v = np.clip((vp[s] - lo) / (hi - lo + 1e-6), 0, 1)
    z = CUBE / side
    img = ndi.zoom(v, z, order=1)[:CUBE, :CUBE, :CUBE]
    msk = ndi.zoom(lp[s].astype(np.float32), z, order=1)[:CUBE, :CUBE, :CUBE] > 0.5
    out = np.zeros((2, CUBE, CUBE, CUBE), np.float16)
    out[0, :img.shape[0], :img.shape[1], :img.shape[2]] = img
    out[1, :msk.shape[0], :msk.shape[1], :msk.shape[2]] = msk
    return out, dict(cube_side_mm=float(side * sp.min()))


# =============================================================================
# Histopathology
# =============================================================================
def hist_tiles(row):
    with Image.open(row.hist_path) as h:
        rgb = np.asarray(h.convert("RGB"), np.float32) / 255.0
    pid = row.patient_id
    lab_path = None
    for ext in (".png", ".jpg", ".tif"):
        p = HIST_LABEL_DIR / f"{pid}_HIST_Label{ext}"
        if p.exists():
            lab_path = p; break
    if lab_path is not None:
        with Image.open(lab_path) as l:
            roi = np.asarray(l.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
    else:
        roi = np.ones(rgb.shape[:2], bool)
    if roi.sum() < 0.01 * roi.size:
        roi = np.ones(rgb.shape[:2], bool)
    H, W = roi.shape
    T = S if min(H, W) >= 2 * S else max(32, min(H, W) // 2)
    cands = []
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            fr = roi[y:y + T, x:x + T].mean()
            if fr >= 0.5:
                cands.append((fr, y, x))
    cands.sort(reverse=True)
    if len(cands) > MAX_TILES:  # spread evenly over the ROI rather than the top block
        sel = np.linspace(0, len(cands) - 1, MAX_TILES).round().astype(int)
        cands = [cands[i] for i in sel]
    tiles = [rs(rgb[y:y + T, x:x + T]) for _, y, x in cands]
    rr, cc = np.where(roi)
    roi_view = rs(rgb[rr.min():rr.max() + 1, cc.min():cc.max() + 1])
    if not tiles:
        tiles = [roi_view]
    return tiles, roi_view, dict(size=f"{W}x{H}", roi_frac=float(roi.mean()), tile_px=T,
                                 n_tiles=len(tiles), has_roi_mask=lab_path is not None)


def path(src, pid):
    return out_dir("features", src) / f"{pid}.npz"


def done(src, pid):
    p = path(src, pid)
    if not p.exists():
        return False
    try:
        with np.load(p) as z:
            return str(z["version"]) == VERSION
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    set_seed(C.SEED)
    df = pd.read_csv(out_dir("cohort") / "cohort_all.csv")
    df = df[df.eligible]
    if args.limit:
        df = df.head(args.limit)
    log(f"CBCT-study features for {len(df)} patients (version {VERSION})")
    enc = Encoders(pretrained=C.PRETRAINED)
    rows = []
    for i, row in enumerate(df.itertuples(), 1):
        pid = row.patient_id
        need_c = not all(done(s, pid) for s in ["RAD", "CBCT3D"] + [f"C25_{b}" for b in BACKBONES])
        need_h = not all(done(f"HT_{b}", pid) for b in BACKBONES)
        if need_c:
            log(f"[{i}/{len(df)}] {pid} CBCT")
            vol, lab, sp, lo, hi, p995 = load_cbct(row)
            if lab.sum() == 0:
                log(f"  WARNING {pid}: empty lesion mask, skipped"); continue
            feats, lab = radiomics(vol, lab, sp, lo, hi, p995)
            names = sorted(feats)
            np.savez(path("RAD", pid), rad=np.array([[feats[k] for k in names]], np.float32),
                     names=np.array(names), version=VERSION)
            views, vmeta = views_25d(vol, lab, sp, lo, hi)
            emb = enc(views)
            for b in BACKBONES:
                np.savez_compressed(path(f"C25_{b}", pid), planes=emb[b],
                                    preview=(np.stack(views[::N_POS]) * 255).astype(np.uint8),
                                    meta=json.dumps(vmeta), version=VERSION)
            cube, cmeta = cube3d(vol, lab, sp, lo, hi)
            np.savez_compressed(path("CBCT3D", pid), cube=cube, meta=json.dumps(cmeta), version=VERSION)
            del vol, lab
        if need_h:
            log(f"[{i}/{len(df)}] {pid} histology tiles")
            tiles, roi_view, hmeta = hist_tiles(row)
            emb = enc(tiles + [roi_view])
            for b in BACKBONES:
                np.savez_compressed(path(f"HT_{b}", pid), tiles=emb[b][:-1], roi=emb[b][-1:],
                                    meta=json.dumps(hmeta), version=VERSION)
        with np.load(path("RAD", pid)) as z:
            r = dict(zip(z["names"], z["rad"][0]))
        with np.load(path("HT_r18", pid)) as z:
            r.update({f"hist_{k}": v for k, v in json.loads(str(z["meta"])).items()})
        with np.load(path("CBCT3D", pid)) as z:
            r.update(json.loads(str(z["meta"])))
        rows.append(dict(patient_id=pid, clinical_class=row.clinical_class, **r))
    pd.DataFrame(rows).to_csv(out_dir("cohort") / "lesion_descriptors.csv", index=False)
    log("Step 6 complete.")


if __name__ == "__main__":
    main()

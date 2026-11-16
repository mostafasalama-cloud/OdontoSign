"""
STEP 12: Semi-automatic end-to-end pipeline (box prompt -> 3D U-Net -> descriptors).

Clinical workflow simulated: the clinician draws a rough 3D box around the
lesion; a 3D U-Net segments the lesion inside the box; the 43 descriptors are
computed from the predicted mask; the descriptor classifier predicts the type.

A. Box prompt (per patient, seeded): expert-mask bounding box, each face moved
   outward by U(2, 8) mm and the centre shifted by up to +-3 mm per axis.
   Cube = box enlarged by 15% (>= 20 mm), resampled to 96^3.
   Channels: CBCT intensity, box mask.  Target: expert lesion mask.
B. Out-of-fold segmentation: stratified 5-fold split (four-class labels, seed 42).
   A U-Net is trained on the training patients' cubes and predicts the
   held-out patients. The U-Net never sees diagnosis labels, so these OOF masks
   can be used by every classification task and CV repeat without leakage.
C. Descriptors from predicted masks (probabilities upsampled to native
   resolution, threshold 0.5, largest component) + segmentation metrics.
D. Classification (10 x 5-fold CV, same splits as before), trained on:
     expert      descriptors from expert masks
     augmented   expert + eroded/dilated (1, 2 mm) masks (step 10 features)
     predicted   descriptors from OOF predicted masks of the training patients
   and always tested on descriptors from the held-out patients' predicted masks.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import time
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi

import config as C
from common import compute_metrics, load_cohort, log, out_dir, pick_device

CUBE = 96
EPOCHS = int(_os.environ.get("DOLCHID_SEG_EPOCHS", "100"))
BATCH = 2
SEG_FOLDS = 5


# =============================================================================
# A. cubes
# =============================================================================
def box_prompt(lab, sp, pid):
    rng = np.random.default_rng(zlib.crc32(pid.encode()))
    idx = np.argwhere(lab)
    lo, hi = idx.min(0).astype(float), idx.max(0).astype(float)
    lo -= rng.uniform(2, 8, 3) / sp
    hi += rng.uniform(2, 8, 3) / sp
    shift = rng.uniform(-3, 3, 3) / sp
    lo, hi = lo + shift, hi + shift
    lo = np.clip(np.floor(lo), 0, np.array(lab.shape) - 1).astype(int)
    hi = np.clip(np.ceil(hi), 0, np.array(lab.shape) - 1).astype(int)
    return lo, hi


def safe_crop(a, start, side, fill):
    out = np.full((side,) * 3, fill, dtype=a.dtype)
    src, dst = [], []
    for st, n in zip(start, a.shape):
        s0, s1 = max(st, 0), min(st + side, n)
        src.append(slice(s0, s1)); dst.append(slice(s0 - st, s1 - st))
    if all(x.stop > x.start for x in src):
        out[tuple(dst)] = a[tuple(src)]
    return out


def prepare_patient(row, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from step6_cbct_features import keep_largest, load_cbct
    from common import out_dir
    p = out_dir("features", "SEG96") / f"{row['patient_id']}.npz"
    if p.exists():
        return
    vol, lab, sp, lo_i, hi_i, p995 = load_cbct(type("R", (), row)())
    lab, _ = keep_largest(lab)
    blo, bhi = box_prompt(lab, sp, row["patient_id"])
    side = int(np.ceil(max((bhi - blo + 1).max() * 1.15, 20 / sp.min())))
    cen = ((blo + bhi) / 2).round().astype(int)
    start = cen - side // 2
    # crop (padding only the small cube, not the whole volume); geometry stored
    v = np.clip((safe_crop(vol, start, side, lo_i) - lo_i) / (hi_i - lo_i + 1e-6), 0, 1).astype(np.float32)
    m = safe_crop(lab, start, side, 0).astype(np.float32)
    del vol
    box = np.zeros_like(m)
    bs = tuple(slice(max(0, b - a), min(side, e - a + 1)) for a, b, e in zip(start, blo, bhi))
    box[bs] = 1
    z = CUBE / side
    rs = lambda a, o: ndi.zoom(a, z, order=o)[:CUBE, :CUBE, :CUBE]
    np.savez_compressed(p, image=rs(v, 1).astype(np.float16), box=(rs(box, 0) > 0.5),
                        mask=(rs(m, 1) > 0.5), start=start, side=side,
                        box_lo=blo, box_hi=bhi, shape=np.array(lab.shape))


# =============================================================================
# B. U-Net
# =============================================================================
def build_unet():
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True),
                             nn.LeakyReLU(0.01, inplace=True),
                             nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True),
                             nn.LeakyReLU(0.01, inplace=True))

    class UNet(nn.Module):
        def __init__(self, ch=(16, 32, 64, 128)):
            super().__init__()
            self.enc = nn.ModuleList()
            self.down = nn.ModuleList()
            c_in = 2
            for c in ch:
                self.enc.append(block(c_in, c)); c_in = c
            for a, b in zip(ch[:-1], ch[1:]):
                self.down.append(nn.Conv3d(a, a, 2, stride=2))
            self.up = nn.ModuleList([nn.ConvTranspose3d(b, a, 2, stride=2) for a, b in zip(ch[:-1], ch[1:])])
            self.dec = nn.ModuleList([block(2 * a, a) for a in ch[:-1]])
            self.out = nn.Conv3d(ch[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x)
                if i < len(self.down):
                    skips.append(x); x = self.down[i](x)
            for i in reversed(range(len(self.up))):
                x = self.up[i](x)
                x = self.dec[i](torch.cat([x, skips[i]], 1))
            return self.out(x)
    return UNet()


def augment(img, box, mask, rng):
    x = np.stack([img, box, mask]).astype(np.float32)
    for ax in (1, 2, 3):
        if rng.random() < 0.5:
            x = np.flip(x, ax)
    x = np.rot90(x, rng.integers(4), axes=(1, 2))
    x = np.ascontiguousarray(x)
    x[0] = np.clip(x[0] * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05) + rng.normal(0, 0.02, x[0].shape), 0, 1)
    k = int(rng.integers(-3, 4))  # box jitter: grow/shrink the prompt box by up to 3 voxels
    if k > 0:
        x[1] = ndi.binary_dilation(x[1] > 0.5, iterations=k)
    elif k < 0:
        x[1] = ndi.binary_erosion(x[1] > 0.5, iterations=-k)
        if x[1].sum() == 0:
            x[1] = x[2] > 0.5
    return x[:2], x[2:3]


def train_predict(data, tr, te, seed, dev):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_unet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    bce = torch.nn.BCEWithLogitsLoss()
    net.train()
    t0 = time.time()
    for ep in range(EPOCHS):
        order = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            xs, ys = zip(*[augment(data["image"][j], data["box"][j], data["mask"][j], rng) for j in b])
            x = torch.from_numpy(np.stack(xs)).to(dev); y = torch.from_numpy(np.stack(ys)).to(dev)
            opt.zero_grad()
            logit = net(x)
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y).sum((1, 2, 3, 4)) + 1) / (p.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4)) + 1)
            loss = bce(logit, y) + dice.mean()
            loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        if ep == 0 or (ep + 1) % 10 == 0:
            log(f"    epoch {ep + 1}/{EPOCHS} loss {tot / max(1, len(order) // BATCH):.4f} ({time.time() - t0:.0f}s)")
    net.eval()
    preds = {}
    with torch.no_grad():
        for j in te:
            x = torch.from_numpy(np.stack([data["image"][j], data["box"][j]]).astype(np.float32)[None]).to(dev)
            p = 0
            for d in (None, [2], [3], [4]):
                xi = torch.flip(x, d) if d else x
                o = torch.sigmoid(net(xi))
                p = p + (torch.flip(o, d) if d else o)
            preds[j] = (p / 4)[0, 0].float().cpu().numpy().astype(np.float16)
    return preds


# =============================================================================
# C. descriptors from predicted masks
# =============================================================================
def hd95_mm(a, b, sp):
    if not a.any() or not b.any():
        return np.nan
    sa = a & ~ndi.binary_erosion(a); sb = b & ~ndi.binary_erosion(b)
    da = ndi.distance_transform_edt(~sb, sampling=sp)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=sp)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def descriptors_from_pred(row, prob, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from step6_cbct_features import keep_largest, load_cbct, radiomics
    from common import out_dir
    p = out_dir("features", "RADS") / f"{row['patient_id']}.npz"
    if p.exists():
        return
    g = np.load(out_dir("features", "SEG96") / f"{row['patient_id']}.npz")
    vol, lab, sp, lo, hi, p995 = load_cbct(type("R", (), row)())
    lab, _ = keep_largest(lab)
    side, start = int(g["side"]), g["start"]
    up = ndi.zoom(prob.astype(np.float32), side / CUBE, order=1)
    up = np.pad(up, [(0, max(0, side - s)) for s in up.shape])[:side, :side, :side] > 0.5
    full = np.zeros(lab.shape, bool)
    src, dst = [], []
    for a, n in zip(start, lab.shape):
        s0, s1 = max(a, 0), min(a + side, n)
        dst.append(slice(s0, s1)); src.append(slice(s0 - a, s1 - a))
    full[tuple(dst)] = up[tuple(src)]
    if full.sum() < 100:   # empty prediction: fall back to the prompt box
        lo_b, hi_b = g["box_lo"], g["box_hi"]
        full[tuple(slice(a, b + 1) for a, b in zip(lo_b, hi_b))] = True
        failed = True
    else:
        failed = False
    full, _ = keep_largest(full)
    inter = (full & lab).sum()
    dice = 2 * inter / (full.sum() + lab.sum())
    # HD95 on a crop around both masks
    idx = np.argwhere(full | lab); a0 = np.maximum(idx.min(0) - 5, 0); a1 = idx.max(0) + 6
    sl = tuple(slice(a, b) for a, b in zip(a0, a1))
    f, _ = radiomics(vol, full, sp, lo, hi, p995)
    names = sorted(f)
    np.savez(p, rad=np.array([[f[k] for k in names]], np.float32), names=np.array(names),
             metrics=json.dumps(dict(dice=float(dice), hd95_mm=hd95_mm(full[sl], lab[sl], sp),
                                     volume_ratio=float(full.sum() / lab.sum()), failed=failed)))


# =============================================================================
# D. classification
# =============================================================================
def classify(task):
    C.TASK = task
    from cbct_study import study_dir
    from step3_evaluate import make_splits
    from step10_cbct_extra import fit_lr, rad_matrix
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    nc = len(C.TASK_CLASSES[task])
    splits = make_splits(y)
    X0, _ = rad_matrix(pids)
    XP = {v: rad_matrix(pids, v)[0] for v in ["erode1", "erode2", "dilate1", "dilate2"]}
    XS = np.stack([np.load(out_dir("features", "RADS") / f"{p}.npz")["rad"][0] for p in pids]).astype(np.float32)
    aggr = [C.TASK_CLASSES[task].index(c) for c in ("AME", "KCOT") if c in C.TASK_CLASSES[task]]

    def run(r):
        rows, oof = [], {k: np.zeros((len(y), nc), np.float32) for k in ("expert", "augmented", "predicted", "expert_on_expert")}
        for f, (tr, te) in enumerate(splits[r]):
            seed = C.SEED + 1000 * r + f
            fits = {"expert": fit_lr(X0[tr], y[tr], seed),
                    "augmented": fit_lr(np.vstack([X0[tr]] + [XP[v][tr] for v in XP]), np.tile(y[tr], 5), seed),
                    "predicted": fit_lr(XS[tr], y[tr], seed)}
            for k, clf in fits.items():
                oof[k][te] = clf.predict_proba(XS[te])
            oof["expert_on_expert"][te] = fits["expert"].predict_proba(X0[te])
        for k, P in oof.items():
            m = compute_metrics(y, P, nc)
            yb = np.isin(y, aggr).astype(int); pb = P[:, aggr].sum(1)
            mb = compute_metrics(yb, np.c_[1 - pb, pb], 2)
            rows.append(dict(repeat=r, train=k, **m, aggr_auc=mb["roc_auc"], aggr_ba=mb["balanced_accuracy"]))
        return rows

    res = pd.DataFrame(sum(Parallel(n_jobs=C.N_JOBS)(delayed(run)(r) for r in range(C.N_REPEATS)), []))
    cols = ["balanced_accuracy", "accuracy", "f1_macro", "roc_auc", "aggr_ba", "aggr_auc"]
    agg = res.groupby("train")[cols].agg(["mean", "std"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.to_csv(study_dir("extra") / "semiauto_classification.csv")
    log(f"[{task}] semi-automatic pipeline (tested on predicted masks; expert_on_expert = reference):\n{agg.to_string()}")


def main():
    df = pd.read_csv(out_dir("cohort") / "cohort_all.csv")
    df = df[df.eligible].reset_index(drop=True)
    recs = df[["patient_id", "cbct_path", "label_path"]].to_dict("records")
    t0 = time.time()
    log(f"A. preparing {len(recs)} box-prompted cubes")
    Parallel(n_jobs=int(_os.environ.get("DOLCHID_EXTRACT_JOBS", "3")))(delayed(prepare_patient)(r) for r in recs)
    log(f"   done ({time.time() - t0:.0f}s)")

    pids = df.patient_id.tolist()
    labels = df.clinical_class.map({c: i for i, c in enumerate(C.TASK_CLASSES["ALL4"])}).to_numpy()
    pred_dir = out_dir("results", "segmentation", "oof_probs")
    todo = [i for i, p in enumerate(pids) if not (pred_dir / f"{p}.npy").exists()]
    if todo:
        import torch
        from sklearn.model_selection import StratifiedKFold
        dev = pick_device(C.FEATURE_DEVICE)
        data = {k: [] for k in ("image", "box", "mask")}
        for p in pids:
            z = np.load(out_dir("features", "SEG96") / f"{p}.npz")
            data["image"].append(z["image"].astype(np.float32)); data["box"].append(z["box"].astype(np.float32))
            data["mask"].append(z["mask"].astype(np.float32))
        folds = list(StratifiedKFold(SEG_FOLDS, shuffle=True, random_state=C.SEED).split(np.zeros(len(pids)), labels))
        for f, (tr, te) in enumerate(folds):
            if all((pred_dir / f"{pids[j]}.npy").exists() for j in te):
                continue
            log(f"B. U-Net fold {f + 1}/{SEG_FOLDS} on {dev}: train {len(tr)}, test {len(te)}, {EPOCHS} epochs")
            try:
                preds = train_predict(data, tr, te, C.SEED + f, dev)
            except RuntimeError as e:
                if dev.type == "cpu":
                    raise
                log(f"  {dev} failed ({str(e)[:150]}); falling back to CPU"); dev = torch.device("cpu")
                preds = train_predict(data, tr, te, C.SEED + f, dev)
            for j, pr in preds.items():
                np.save(pred_dir / f"{pids[j]}.npy", pr)
            d96 = [2 * ((preds[j] > 0.5) & (data["mask"][j] > 0.5)).sum() /
                   ((preds[j] > 0.5).sum() + (data["mask"][j] > 0.5).sum() + 1e-6) for j in te]
            log(f"   fold {f + 1}: median Dice (96^3 grid) {np.median(d96):.3f}")
        json.dump(dict(epochs=EPOCHS, cube=CUBE, folds=SEG_FOLDS, seed=C.SEED),
                  open(out_dir("results", "segmentation") / "seg_config.json", "w"))

    log("C. descriptors from predicted masks")
    Parallel(n_jobs=int(_os.environ.get("DOLCHID_EXTRACT_JOBS", "3")))(
        delayed(descriptors_from_pred)(r, np.load(pred_dir / f"{r['patient_id']}.npy")) for r in recs)
    rows = []
    for r in recs:
        z = np.load(out_dir("features", "RADS") / f"{r['patient_id']}.npz")
        rows.append(dict(patient_id=r["patient_id"], clinical_class=r["patient_id"].split("_")[0],
                         **json.loads(str(z["metrics"]))))
    seg = pd.DataFrame(rows)
    seg.to_csv(out_dir("results", "segmentation") / "segmentation_metrics.csv", index=False)
    summ = seg.groupby("clinical_class")[["dice", "hd95_mm", "volume_ratio"]].median().round(3)
    summ.loc["ALL"] = seg[["dice", "hd95_mm", "volume_ratio"]].median().round(3)
    summ["n_failed"] = seg.groupby("clinical_class").failed.sum().reindex(summ.index).fillna(seg.failed.sum())
    summ.to_csv(out_dir("results", "segmentation") / "segmentation_summary.csv")
    log(f"Segmentation (median per class):\n{summ.to_string()}")

    for task in ["ALL4", "AME_vs_DC"]:
        classify(task)
    log("Step 12 complete.")


if __name__ == "__main__":
    main()

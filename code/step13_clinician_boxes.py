"""
STEP 13: Confirmatory test of OdontoSign with boxes drawn by a clinician.

60 patients (15 per class) were anonymised and given to a clinician, who drew one
3D box per scan in 3D Slicer (box_study/boxes_clinician/case_XX_box.mrk.json).
Everything else is frozen; the settings below are those of step 12.

A. Clinician boxes -> voxel boxes (native grid, RAS+ as in load_cbct) -> 96^3 cubes
   built exactly like the simulated-box cubes (SEG96C/<pid>.npz).
B. The box-prompted U-Net is retrained with the step-12 settings and the same
   5-fold split and seeds (weights saved this time). Each fold model predicts its
   held-out patients; for the 60 study patients it predicts both from the
   clinician box and from the simulated box (paired comparison, same network).
C. Descriptors from both predictions (RADC = clinician, RADS2 = simulated).
D. Classification: the frozen deployment model (logistic regression trained on
   the U-Net descriptors of the training patients, same 10 x 5 CV splits) is
   applied to the held-out study patients with each type of input.
   Pre-specified hypothesis: four-class macro-AUC with clinician boxes > 0.801
   (best published CBCT model); 95% stratified bootstrap CI and one-sided P.
Outputs: results/clinician_boxes/*.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi

import config as C
from common import compute_metrics, load_cohort, log, out_dir, pick_device
from step12_semiauto_seg import (BATCH, CUBE, EPOCHS, SEG_FOLDS, augment, build_unet,
                                 hd95_mm, safe_crop)

STUDY = _os.environ.get("DOLCHID_READER_STUDY",
                        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "reader_study"))
BENCHMARK_AUC = 0.801
N_BOOT = 2000


# =============================================================================
# A. clinician boxes -> cubes
# =============================================================================
def clinician_box_voxels(cbct_path, box_json):
    import nibabel as nib
    m = json.load(open(box_json))["markups"][0]
    c = np.array(m["center"], float); s = np.array(m["size"], float)
    R = np.array(m.get("orientation", np.eye(3).ravel()), float).reshape(3, 3)
    if m.get("coordinateSystem", "LPS") == "LPS":
        F = np.diag([-1.0, -1.0, 1.0]); c = F @ c; R = F @ R
    corners = np.array([[a, b, d] for a in (-.5, .5) for b in (-.5, .5) for d in (-.5, .5)]) * s
    ras = c + corners @ R.T
    im = nib.load(cbct_path)
    ijk = (np.linalg.inv(im.affine) @ np.c_[ras, np.ones(8)].T).T[:, :3]
    # same reorientation as load_cbct (io_orientation + apply_orientation)
    ornt = nib.orientations.io_orientation(im.affine)
    shp = np.array(im.shape[:3])
    out = np.empty_like(ijk)
    for ax_in in range(3):
        ax_out, flip = int(ornt[ax_in, 0]), ornt[ax_in, 1]
        out[:, ax_out] = ijk[:, ax_in] if flip > 0 else (shp[ax_in] - 1 - ijk[:, ax_in])
    new_shape = np.empty(3, int); new_shape[ornt[:, 0].astype(int)] = shp
    lo = np.clip(np.floor(out.min(0)), 0, new_shape - 1).astype(int)
    hi = np.clip(np.ceil(out.max(0)), 0, new_shape - 1).astype(int)
    return lo, hi


def prepare_clinician(row, box_json, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from step6_cbct_features import keep_largest, load_cbct
    from common import out_dir
    p = out_dir("features", "SEG96C") / f"{row['patient_id']}.npz"
    if p.exists():
        return
    vol, lab, sp, lo_i, hi_i, p995 = load_cbct(type("R", (), row)())
    lab, _ = keep_largest(lab)
    blo, bhi = clinician_box_voxels(row["cbct_path"], box_json)
    side = int(np.ceil(max((bhi - blo + 1).max() * 1.15, 20 / sp.min())))
    cen = ((blo + bhi) / 2).round().astype(int)
    start = cen - side // 2
    v = np.clip((safe_crop(vol, start, side, lo_i) - lo_i) / (hi_i - lo_i + 1e-6), 0, 1).astype(np.float32)
    m = safe_crop(lab, start, side, 0).astype(np.float32)
    del vol
    box = np.zeros_like(m)
    bs = tuple(slice(max(0, b - a), min(side, e - a + 1)) for a, b, e in zip(start, blo, bhi))
    box[bs] = 1
    z = CUBE / side
    rs = lambda a, o: ndi.zoom(a, z, order=o)[:CUBE, :CUBE, :CUBE]
    inside = float(lab[tuple(slice(a, b + 1) for a, b in zip(blo, bhi))].sum() / max(lab.sum(), 1))
    np.savez_compressed(p, image=rs(v, 1).astype(np.float16), box=(rs(box, 0) > 0.5),
                        mask=(rs(m, 1) > 0.5), start=start, side=side,
                        box_lo=blo, box_hi=bhi, shape=np.array(lab.shape), lesion_in_box=inside)


# =============================================================================
# B. U-Net (step-12 recipe, weights saved)
# =============================================================================
def train_save_predict(data, tr, te, seed, dev, path):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build_unet().to(dev)
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=dev))
        log(f"    loaded saved weights {path.name}")
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
        bce = torch.nn.BCEWithLogitsLoss()
        net.train(); t0 = time.time()
        for ep in range(EPOCHS):
            order = rng.permutation(tr); tot = 0.0
            for i in range(0, len(order), BATCH):
                b = order[i:i + BATCH]
                xs, ys = zip(*[augment(data["image"][j], data["box"][j], data["mask"][j], rng) for j in b])
                x = torch.from_numpy(np.stack(xs)).to(dev); y = torch.from_numpy(np.stack(ys)).to(dev)
                opt.zero_grad()
                logit = net(x); pr = torch.sigmoid(logit)
                dice = 1 - (2 * (pr * y).sum((1, 2, 3, 4)) + 1) / (pr.sum((1, 2, 3, 4)) + y.sum((1, 2, 3, 4)) + 1)
                loss = bce(logit, y) + dice.mean()
                loss.backward(); opt.step(); sched.step()
                tot += float(loss.detach())
            if ep == 0 or (ep + 1) % 10 == 0:
                log(f"    epoch {ep + 1}/{EPOCHS} loss {tot / max(1, len(order) // BATCH):.4f} ({time.time() - t0:.0f}s)")
        torch.save(net.state_dict(), path)
    net.eval()
    preds = {}
    with torch.no_grad():
        for key, (img, box) in te.items():
            x = torch.from_numpy(np.stack([img, box]).astype(np.float32)[None]).to(dev)
            p = 0
            for d in (None, [2], [3], [4]):
                xi = torch.flip(x, d) if d else x
                o = torch.sigmoid(net(xi))
                p = p + (torch.flip(o, d) if d else o)
            preds[key] = (p / 4)[0, 0].float().cpu().numpy().astype(np.float16)
    return preds


# =============================================================================
# C. descriptors
# =============================================================================
def descriptors(row, prob, geom_dir, out_name, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from step6_cbct_features import keep_largest, load_cbct, radiomics
    from step12_semiauto_seg import hd95_mm, CUBE
    from common import out_dir
    p = out_dir("features", out_name) / f"{row['patient_id']}.npz"
    if p.exists():
        return
    g = np.load(out_dir("features", geom_dir) / f"{row['patient_id']}.npz")
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
    failed = bool(full.sum() < 100)
    if failed:
        full[tuple(slice(a, b + 1) for a, b in zip(g["box_lo"], g["box_hi"]))] = True
    full, _ = keep_largest(full)
    dice = 2 * (full & lab).sum() / (full.sum() + lab.sum())
    idx = np.argwhere(full | lab); a0 = np.maximum(idx.min(0) - 5, 0); a1 = idx.max(0) + 6
    sl = tuple(slice(a, b) for a, b in zip(a0, a1))
    f, _ = radiomics(vol, full, sp, lo, hi, p995)
    names = sorted(f)
    np.savez(p, rad=np.array([[f[k] for k in names]], np.float32), names=np.array(names),
             metrics=json.dumps(dict(dice=float(dice), hd95_mm=hd95_mm(full[sl], lab[sl], sp),
                                     volume_ratio=float(full.sum() / lab.sum()), failed=failed)))


# =============================================================================
# D. classification on the study patients
# =============================================================================
def boot_ci(y, P, nc, metric, rng, n=N_BOOT):
    vals = []
    classes = np.unique(y)
    for _ in range(n):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], (y == c).sum(), replace=True) for c in classes])
        vals.append(compute_metrics(y[idx], P[idx], nc)[metric])
    return np.percentile(vals, [2.5, 97.5]), np.array(vals)


def classify(task, study_pids):
    C.TASK = task
    from step3_evaluate import make_splits
    from step10_cbct_extra import fit_lr, rad_matrix
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    nc = len(C.TASK_CLASSES[task])
    splits = make_splits(y)
    sidx = np.array([i for i, p in enumerate(pids) if p in study_pids])
    load = lambda d: np.stack([np.load(out_dir("features", d) / f"{pids[i]}.npz")["rad"][0] for i in sidx]).astype(np.float32)
    X0, _ = rad_matrix(pids)
    XS = np.stack([np.load(out_dir("features", "RADS") / f"{p}.npz")["rad"][0] for p in pids]).astype(np.float32)
    XIN = {"clinician_box": load("RADC"), "simulated_box": load("RADS2"), "expert_mask": X0[sidx]}
    aggr = [C.TASK_CLASSES[task].index(c) for c in ("AME", "KCOT") if c in C.TASK_CLASSES[task]]
    ys = y[sidx]
    pos = {i: k for k, i in enumerate(sidx)}

    rows, P_all = [], {k: np.zeros((C.N_REPEATS, len(sidx), nc)) for k in XIN}
    for r in range(C.N_REPEATS):
        for f, (tr, te) in enumerate(splits[r]):
            seed = C.SEED + 1000 * r + f
            te_s = [i for i in te if i in pos]
            if not te_s:
                continue
            k_idx = [pos[i] for i in te_s]
            clf_pred = fit_lr(XS[tr], y[tr], seed)         # deployment model (frozen recipe)
            clf_exp = fit_lr(X0[tr], y[tr], seed)          # expert-mask reference
            P_all["clinician_box"][r, k_idx] = clf_pred.predict_proba(XIN["clinician_box"][k_idx])
            P_all["simulated_box"][r, k_idx] = clf_pred.predict_proba(XIN["simulated_box"][k_idx])
            P_all["expert_mask"][r, k_idx] = clf_exp.predict_proba(XIN["expert_mask"][k_idx])
        for k in XIN:
            m = compute_metrics(ys, P_all[k][r], nc)
            row = dict(repeat=r, input=k, **{a: m[a] for a in ("balanced_accuracy", "accuracy", "f1_macro", "roc_auc")})
            if aggr and nc > 2:
                yb = np.isin(ys, aggr).astype(int); pb = P_all[k][r][:, aggr].sum(1)
                row["aggr_auc"] = compute_metrics(yb, np.c_[1 - pb, pb], 2)["roc_auc"]
            rows.append(row)
    res = pd.DataFrame(rows)
    cols = [c for c in ("balanced_accuracy", "accuracy", "f1_macro", "roc_auc", "aggr_auc") if c in res]
    agg = res.groupby("input")[cols].agg(["mean", "std"]).round(3)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]

    # repeat-averaged probabilities -> bootstrap CIs, paired difference, hypothesis test
    rng = np.random.default_rng(C.SEED)
    Pm = {k: P_all[k].mean(0) for k in XIN}
    ci_rows = []
    for k in XIN:
        for met in ("roc_auc", "balanced_accuracy"):
            (lo, hi), _ = boot_ci(ys, Pm[k], nc, met, np.random.default_rng(C.SEED))
            ci_rows.append(dict(input=k, metric=met, point=round(compute_metrics(ys, Pm[k], nc)[met], 3),
                                ci_low=round(lo, 3), ci_high=round(hi, 3)))
    ci = pd.DataFrame(ci_rows)
    # paired bootstrap: clinician - simulated (AUC)
    d = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(np.where(ys == c)[0], (ys == c).sum(), replace=True) for c in np.unique(ys)])
        d.append(compute_metrics(ys[idx], Pm["clinician_box"][idx], nc)["roc_auc"] -
                 compute_metrics(ys[idx], Pm["simulated_box"][idx], nc)["roc_auc"])
    d = np.array(d)
    od = out_dir("results", "clinician_boxes")
    summary = dict(task=task, n=int(len(sidx)),
                   delta_auc_clinician_minus_simulated=round(float(np.mean(d)), 3),
                   delta_ci=[round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])])
    if task == "ALL4":
        _, bs = boot_ci(ys, Pm["clinician_box"], nc, "roc_auc", np.random.default_rng(C.SEED + 1))
        summary.update(hypothesis="macro-AUC(clinician boxes) > %.3f" % BENCHMARK_AUC,
                       auc=round(compute_metrics(ys, Pm["clinician_box"], nc)["roc_auc"], 3),
                       one_sided_p=round(float((1 + (bs <= BENCHMARK_AUC).sum()) / (len(bs) + 1)), 4))
    agg.to_csv(od / f"classification_{task}.csv")
    ci.to_csv(od / f"bootstrap_ci_{task}.csv", index=False)
    json.dump(summary, open(od / f"summary_{task}.json", "w"), indent=1)
    log(f"[{task}] study patients n={len(sidx)} (mean +- SD over {C.N_REPEATS} repeats):\n{agg.to_string()}")
    log(f"[{task}] bootstrap 95% CIs (repeat-averaged probabilities):\n{ci.to_string(index=False)}")
    log(f"[{task}] {json.dumps(summary)}")


# =============================================================================
def main():
    key = pd.read_csv(_os.path.join(STUDY, "case_key.csv"))
    bdir = _os.path.join(STUDY, "boxes_clinician")
    df = pd.read_csv(out_dir("cohort") / "cohort_all.csv")
    df = df[df.eligible].reset_index(drop=True)
    recs = {r["patient_id"]: r for r in df[["patient_id", "cbct_path", "label_path"]].to_dict("records")}
    study = {row.patient_id: _os.path.join(bdir, f"{row.case}_box.mrk.json") for row in key.itertuples()}
    missing = [p for p, f in study.items() if not _os.path.exists(f)]
    if missing:
        sys_exit = f"Missing clinician boxes for {missing}"
        raise SystemExit(sys_exit)
    plan = out_dir("results", "clinician_boxes") / "analysis_plan.json"
    if not plan.exists():   # written once, before any segmentation or classification is run
        json.dump(dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), n_patients=len(study),
                       segmentation="step-12 U-Net recipe retrained with identical settings/split/seeds",
                       classifier="L2 logistic regression trained on U-Net (simulated-box) descriptors of training folds",
                       primary_endpoint="four-class macro-AUC on the 60 study patients with clinician boxes",
                       hypothesis=f"macro-AUC > {BENCHMARK_AUC} (best published CBCT model)",
                       test="one-sided stratified bootstrap (2000) on repeat-averaged probabilities",
                       secondary=["BA, accuracy, F1", "aggressive-lesion AUC", "AME vs DC",
                                  "paired difference clinician - simulated boxes", "Dice/HD95 clinician vs simulated"]),
                  open(plan, "w"), indent=1)
    log(f"A. {len(study)} clinician boxes -> cubes")
    Parallel(n_jobs=int(_os.environ.get("DOLCHID_EXTRACT_JOBS", "3")))(
        delayed(prepare_clinician)(recs[p], f) for p, f in study.items())
    inside = [float(np.load(out_dir("features", "SEG96C") / f"{p}.npz")["lesion_in_box"]) for p in study]
    log(f"   lesion fraction inside clinician box: median {np.median(inside):.3f}, min {np.min(inside):.3f}")

    pids = df.patient_id.tolist()
    labels = df.clinical_class.map({c: i for i, c in enumerate(C.TASK_CLASSES["ALL4"])}).to_numpy()
    pc = out_dir("results", "clinician_boxes", "probs_clinician")
    ps = out_dir("results", "clinician_boxes", "probs_simulated")
    mdir = out_dir("results", "segmentation", "models")
    if not all((pc / f"{p}.npy").exists() and (ps / f"{p}.npy").exists() for p in study):
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
            te_study = [pids[j] for j in te if pids[j] in study]
            if not te_study or all((pc / f"{p}.npy").exists() for p in te_study):
                continue
            jobs = {}
            for p in te_study:
                j = pids.index(p)
                zc = np.load(out_dir("features", "SEG96C") / f"{p}.npz")
                jobs[("sim", p)] = (data["image"][j], data["box"][j])
                jobs[("clin", p)] = (zc["image"].astype(np.float32), zc["box"].astype(np.float32))
            log(f"B. U-Net fold {f + 1}/{SEG_FOLDS} on {dev}: train {len(tr)}, study patients in test {len(te_study)}")
            try:
                preds = train_save_predict(data, tr, jobs, C.SEED + f, dev, mdir / f"unet_fold{f + 1}.pt")
            except RuntimeError as e:
                if dev.type == "cpu":
                    raise
                log(f"  {dev} failed ({str(e)[:150]}); falling back to CPU"); dev = torch.device("cpu")
                preds = train_save_predict(data, tr, jobs, C.SEED + f, dev, mdir / f"unet_fold{f + 1}.pt")
            for (kind, p), pr in preds.items():
                np.save((pc if kind == "clin" else ps) / f"{p}.npy", pr)
        del data

    log("C. descriptors from clinician-box and simulated-box predictions")
    nj = int(_os.environ.get("DOLCHID_EXTRACT_JOBS", "3"))
    Parallel(n_jobs=nj)(delayed(descriptors)(recs[p], np.load(pc / f"{p}.npy"), "SEG96C", "RADC") for p in study)
    Parallel(n_jobs=nj)(delayed(descriptors)(recs[p], np.load(ps / f"{p}.npy"), "SEG96", "RADS2") for p in study)
    rows = []
    for p in study:
        for kind, d in (("clinician_box", "RADC"), ("simulated_box", "RADS2")):
            z = np.load(out_dir("features", d) / f"{p}.npz")
            rows.append(dict(patient_id=p, clinical_class=p.split("_")[0], input=kind, **json.loads(str(z["metrics"]))))
    seg = pd.DataFrame(rows)
    od = out_dir("results", "clinician_boxes")
    seg.to_csv(od / "segmentation_metrics.csv", index=False)
    summ = seg.groupby("input")[["dice", "hd95_mm", "volume_ratio"]].describe(percentiles=[.25, .5, .75]).round(3)
    summ.to_csv(od / "segmentation_summary.csv")
    from scipy.stats import wilcoxon
    a = seg[seg.input == "clinician_box"].set_index("patient_id").dice
    b = seg[seg.input == "simulated_box"].set_index("patient_id").dice.reindex(a.index)
    log("Segmentation, median [IQR] Dice: clinician %.3f [%.3f-%.3f], simulated %.3f [%.3f-%.3f]; Wilcoxon P=%.3g; failures clinician %d"
        % (a.median(), a.quantile(.25), a.quantile(.75), b.median(), b.quantile(.25), b.quantile(.75),
           wilcoxon(a, b).pvalue, int(seg[seg.input == "clinician_box"].failed.sum())))

    for task in ["ALL4", "AME_vs_DC"]:
        classify(task, set(study))
    json.dump(dict(epochs=EPOCHS, cube=CUBE, folds=SEG_FOLDS, seed=C.SEED, benchmark_auc=BENCHMARK_AUC,
                   n_boot=N_BOOT), open(od / "run_config.json", "w"))
    log("Step 13 complete.")


if __name__ == "__main__":
    main()

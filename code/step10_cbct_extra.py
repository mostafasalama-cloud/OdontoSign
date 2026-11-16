"""
STEP 10 (CBCT-centred study): additional analyses requested for submission.

A. Mask robustness. 3D lesion descriptors are recomputed from perturbed
   lesion masks (exact Euclidean erosion/dilation by 1 and 2 mm, and the
   axis-aligned bounding box). The descriptor model is trained on expert
   masks inside each training fold and applied to perturbed masks of the
   held-out patients (simulating imperfect segmentation at deployment); a
   model trained AND tested on bounding boxes is also reported.
B. Interpretability. Grouped out-of-fold permutation importance of descriptor
   families; univariate Kruskal-Wallis / Mann-Whitney tests with BH-FDR and
   per-class medians [IQR]; standardised multinomial coefficients.
C. Calibration (Brier score, top-label ECE, reliability curves) and decision
   curve analysis for the clinically derived endpoint "aggressive lesion"
   (AME or KCOT, requiring resection / adjunctive therapy) vs cyst (DC, RC).
D. Fold-averaged metrics (mean ± SD over folds) for like-for-like comparison
   with the published DOLCHID benchmark, which reports fold means.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import config as C
from common import compute_metrics, fast_ba, load_cohort, log, out_dir

VARIANTS = ["erode1", "erode2", "dilate1", "dilate2", "bbox"]
GROUPS = {
    "size": ["volume_mm3", "log_volume", "surface_mm2", "axis_major_mm", "axis_minor_mm", "axis_least_mm"],
    "shape": ["sphericity", "solidity", "elongation", "flatness", "extent", "n_mask_components"],
    "internal density": ["int_mean", "int_sd", "int_skew", "int_kurt", "int_p10", "int_p25", "int_p50",
                         "int_p75", "int_p90", "int_entropy"],
    "rim / boundary": ["rim_mean", "rim_sd", "outer_mean", "rim_contrast", "frac_inside_above_rim",
                       "boundary_grad_mean", "boundary_grad_sd"],
    "hyperdense content": ["hyperdense_frac_in", "hyperdense_frac_rim", "hyperdense_mm3_rim",
                           "hyperdense_n_objects"],
    "texture": ["glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
                "glcm_correlation", "glcm_ASM"],
    "location": ["loc_x", "loc_y", "loc_z", "loc_abs_x"],
}


# =============================================================================
# A. perturbed-mask descriptors (label-free, per patient)
# =============================================================================
def perturb(lab, sp, variant):
    idx = np.argwhere(lab)
    pad = np.ceil(15 / sp).astype(int)
    b0 = np.maximum(idx.min(0) - pad, 0); b1 = np.minimum(idx.max(0) + pad + 1, lab.shape)
    sl = tuple(slice(a, b) for a, b in zip(b0, b1))
    m = lab[sl]
    if variant == "bbox":
        i2 = np.argwhere(m); new = np.zeros_like(m)
        new[tuple(slice(a, b + 1) for a, b in zip(i2.min(0), i2.max(0)))] = True
        used = 0.0
    else:
        r = float(variant[-1])
        if variant.startswith("dilate"):
            new = ndi.distance_transform_edt(~m, sampling=sp) <= r
            used = r
        else:
            dt = ndi.distance_transform_edt(m, sampling=sp)
            used = r
            # never erode a lesion below 100 voxels (small lesions): shrink the radius instead
            while used > 0 and (dt > used).sum() < 100:
                used = round(used - 0.3, 2)
            new = dt > used if used > 0 else m
    out = np.zeros_like(lab)
    out[sl] = new
    return out, used


def extract_patient(row, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    import numpy as np
    from step6_cbct_features import keep_largest, load_cbct, radiomics
    from common import out_dir
    p = out_dir("features", "RADP") / f"{row['patient_id']}.npz"
    if p.exists():
        return row["patient_id"], "cached"
    vol, lab, sp, lo, hi, p995 = load_cbct(type("R", (), row)())
    lab, _ = keep_largest(lab)
    res, info = {}, {}
    for v in VARIANTS:
        m, used = perturb(lab, sp, v)
        try:
            f, _ = radiomics(vol, m, sp, lo, hi, p995)
        except Exception:  # degenerate perturbed mask: fall back to the expert mask
            m, used = lab, -1.0
            f, _ = radiomics(vol, m, sp, lo, hi, p995)
        names = sorted(f)
        res[v] = np.array([[f[k] for k in names]], np.float32)
        info[v] = dict(radius_used_mm=used, dice_vs_expert=float(2 * (m & lab).sum() / (m.sum() + lab.sum())))
    np.savez(p, names=np.array(names), info=json.dumps(info), **res)
    return row["patient_id"], "done"


# =============================================================================
# helpers
# =============================================================================
def rad_matrix(pids, variant=None):
    rows, names = [], None
    for pid in pids:
        if variant is None:
            z = np.load(out_dir("features", "RAD") / f"{pid}.npz")
            rows.append(z["rad"][0]); names = list(z["names"])
        else:
            z = np.load(out_dir("features", "RADP") / f"{pid}.npz")
            rows.append(z[variant][0]); names = list(z["names"])
    return np.array(rows, np.float32), names


def fit_lr(X, y, seed):
    from cbct_study import _lr_pipeline
    clf = _lr_pipeline(seed, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def ece_toplabel(y, p, bins=10):
    conf = p.max(1); pred = p.argmax(1); acc = pred == y
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def net_benefit(y, p, t):
    n = len(y); pos = p >= t
    tp = (pos & (y == 1)).sum(); fp = (pos & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)


# =============================================================================
# per-task analyses
# =============================================================================
def analyse(task):
    C.TASK = task
    from cbct_study import study_dir
    from step3_evaluate import make_splits
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    classes = C.TASK_CLASSES[task]; nc = len(classes)
    splits = make_splits(y)
    od = study_dir("extra")
    X0, names = rad_matrix(pids)
    log(f"[{task}] {len(y)} patients, {X0.shape[1]} descriptors")

    # ---------------------------------------------------------------- A
    XV = {v: rad_matrix(pids, v)[0] for v in VARIANTS}
    def run_rob(r):
        rows = []
        for f, (tr, te) in enumerate(splits[r]):
            seed = C.SEED + 1000 * r + f
            clf = fit_lr(X0[tr], y[tr], seed)
            rows.append(dict(repeat=r, fold=f, test="expert", train="expert",
                             **compute_metrics(y[te], clf.predict_proba(X0[te]), nc)))
            for v in VARIANTS:
                rows.append(dict(repeat=r, fold=f, test=v, train="expert",
                                 **compute_metrics(y[te], clf.predict_proba(XV[v][te]), nc)))
            clf_b = fit_lr(XV["bbox"][tr], y[tr], seed)
            rows.append(dict(repeat=r, fold=f, test="bbox", train="bbox",
                             **compute_metrics(y[te], clf_b.predict_proba(XV["bbox"][te]), nc)))
        return rows
    rob = pd.DataFrame(sum(Parallel(n_jobs=C.N_JOBS)(delayed(run_rob)(r) for r in range(C.N_REPEATS)), []))
    dice = {v: [] for v in VARIANTS}
    for pid in pids:
        info = json.loads(str(np.load(out_dir("features", "RADP") / f"{pid}.npz")["info"]))
        for v in VARIANTS:
            dice[v].append(info[v]["dice_vs_expert"])
    rs = (rob.groupby(["train", "test"])[["balanced_accuracy", "accuracy", "f1_macro", "roc_auc"]]
          .agg(["mean", "std"]).round(3))
    rs.columns = [f"{a}_{b}" for a, b in rs.columns]
    rs = rs.reset_index()
    rs["median_dice_vs_expert"] = rs.test.map(lambda v: round(float(np.median(dice[v])), 3) if v in dice else 1.0)
    rs.to_csv(od / "mask_robustness.csv", index=False)
    log(f"[{task}] mask robustness:\n{rs.to_string(index=False)}")

    # ---------------------------------------------------------------- B
    gidx = {g: [names.index(c) for c in cols if c in names] for g, cols in GROUPS.items()}
    def run_imp(r):
        rng = np.random.default_rng(C.SEED + r)
        rows = []
        for f, (tr, te) in enumerate(splits[r]):
            clf = fit_lr(X0[tr], y[tr], C.SEED + 1000 * r + f)
            base = fast_ba(y[te], clf.predict(X0[te]), nc)
            for g, ix in gidx.items():
                drops = []
                for _ in range(20):
                    Xp = X0[te].copy()
                    perm = rng.permutation(len(te))
                    Xp[:, ix] = Xp[perm][:, ix]
                    drops.append(base - fast_ba(y[te], clf.predict(Xp), nc))
                rows.append(dict(repeat=r, fold=f, group=g, ba_drop=float(np.mean(drops))))
        return rows
    imp = pd.DataFrame(sum(Parallel(n_jobs=C.N_JOBS)(delayed(run_imp)(r) for r in range(C.N_REPEATS)), []))
    imp_s = imp.groupby("group").ba_drop.agg(["mean", "std"]).sort_values("mean", ascending=False).round(4)
    imp_s["n_descriptors"] = [len(gidx[g]) for g in imp_s.index]
    imp_s.to_csv(od / "group_importance.csv")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(imp_s.index[::-1], imp_s["mean"][::-1], xerr=imp_s["std"][::-1], color="#3b7dd8")
    ax.set_xlabel("drop in balanced accuracy when permuted"); ax.set_title(f"Descriptor families ({task})")
    fig.tight_layout(); fig.savefig(od / "fig_group_importance.png", dpi=200); plt.close(fig)

    uni = []
    for j, n in enumerate(names):
        v = X0[:, j]
        groups = [v[y == c] for c in range(nc)]
        try:
            if nc == 2:
                stat, p = mannwhitneyu(*groups); eff = stat / (len(groups[0]) * len(groups[1]))
            else:
                stat, p = kruskal(*groups); eff = (stat - nc + 1) / (len(v) - nc)
        except ValueError:
            p, eff = 1.0, 0.0
        row = dict(descriptor=n, family=next((g for g, cols in GROUPS.items() if n in cols), "other"),
                   p=p, effect=eff)
        for c, cl in enumerate(classes):
            q = np.percentile(groups[c], [25, 50, 75])
            row[cl] = f"{q[1]:.3g} [{q[0]:.3g}–{q[2]:.3g}]"
        uni.append(row)
    uni = pd.DataFrame(uni)
    uni["q_fdr"] = multipletests(uni.p, method="fdr_bh")[1]
    uni["effect_name"] = "AUC (Mann-Whitney)" if nc == 2 else "epsilon^2 (Kruskal-Wallis)"
    uni.sort_values("p").to_csv(od / "univariate_descriptors.csv", index=False)

    clf = fit_lr(X0, y, C.SEED)
    coef = clf[-1].coef_
    cdf = pd.DataFrame(coef.T, index=names,
                       columns=[f"coef_{c}" for c in (classes if coef.shape[0] == nc else [classes[1]])])
    cdf.round(4).to_csv(od / "standardised_coefficients_full_data.csv")

    # ---------------------------------------------------------------- C
    oof = pd.read_csv(study_dir() / "oof_predictions.csv.gz")
    pc = [f"prob_{c}" for c in range(nc)]
    cal, curves = [], {}
    for rep in ["RAD", "RAD+C25_dinob", "HT_dinob", "MM_RAD+C25+HT"]:
        g = oof[oof.representation == rep]
        if not len(g):
            continue
        br, ec = [], []
        for _, gr in g.groupby("repeat"):
            P = gr[pc].to_numpy(); yy = gr.label.to_numpy()
            br.append(np.mean(np.sum((P - np.eye(nc)[yy]) ** 2, 1)))
            ec.append(ece_toplabel(yy, P))
        cal.append(dict(model=rep, brier_mean=np.mean(br), brier_sd=np.std(br), ece_mean=np.mean(ec), ece_sd=np.std(ec)))
        P = g[pc].to_numpy(); yy = g.label.to_numpy()
        curves[rep] = (P, yy)
    pd.DataFrame(cal).round(4).to_csv(od / "calibration.csv", index=False)
    if "RAD" in curves:
        P, yy = curves["RAD"]
        fig, ax = plt.subplots(figsize=(4, 4))
        for c, cl in enumerate(classes if nc > 2 else classes[1:]):
            c = c if nc > 2 else 1
            edges = np.linspace(0, 1, 11); mids, obs = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (P[:, c] > a) & (P[:, c] <= b)
                if m.sum() >= 10:
                    mids.append(P[m, c].mean()); obs.append((yy[m] == c).mean())
            ax.plot(mids, obs, "o-", ms=3, label=cl)
        ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency"); ax.legend(fontsize=7); ax.set_title(f"Reliability, descriptors ({task})")
        fig.tight_layout(); fig.savefig(od / "fig_reliability.png", dpi=200); plt.close(fig)

    # decision curves: endpoint "aggressive" = AME (+ KCOT for 4-class)
    aggr = [classes.index(c) for c in ("AME", "KCOT") if c in classes]
    th = np.arange(0.05, 0.96, 0.05)
    dca, endpoint = [], []
    for rep in ["RAD", "RAD+C25_dinob", "HT_dinob"]:
        if rep not in curves:
            continue
        g = oof[oof.representation == rep]
        nb = []
        for _, gr in g.groupby("repeat"):
            yb = gr.label.isin(aggr).astype(int).to_numpy()
            pb = gr[[f"prob_{c}" for c in aggr]].sum(axis=1).to_numpy()
            nb.append([net_benefit(yb, pb, t) for t in th])
            m = compute_metrics(yb, np.c_[1 - pb, pb], 2)
            endpoint.append(dict(model=rep, **{k: m[k] for k in ("balanced_accuracy", "roc_auc", "sensitivity", "specificity")}))
        dca.append((rep, np.mean(nb, 0)))
    yb = np.isin(y, aggr).astype(int); prev = yb.mean()
    treat_all = prev - (1 - prev) * th / (1 - th)
    pd.DataFrame(endpoint).groupby("model").agg(["mean", "std"]).round(3).to_csv(od / "aggressive_endpoint.csv")
    dtab = pd.DataFrame({"threshold": th, "treat_all": treat_all, "treat_none": 0.0,
                         **{rep: nbv for rep, nbv in dca}}).round(4)
    dtab.to_csv(od / "decision_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for rep, nbv in dca:
        ax.plot(th, nbv, label=rep)
    ax.plot(th, treat_all, "k--", lw=1, label="treat all"); ax.axhline(0, c="k", lw=1, label="treat none")
    ax.set_ylim(-0.05, prev + 0.05); ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit"); ax.legend(fontsize=7)
    ax.set_title("Aggressive lesion (" + "+".join(classes[i] for i in aggr) + ")")
    fig.tight_layout(); fig.savefig(od / "fig_decision_curve.png", dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- D
    folds = pd.read_csv(study_dir() / "fold_metrics.csv")
    fa = (folds.groupby("representation")[["accuracy", "f1_macro", "roc_auc", "balanced_accuracy"]]
          .agg(["mean", "std"]) * 100).round(1)
    fa.columns = [f"{a}_{b}_%" for a, b in fa.columns]
    fa.to_csv(od / "fold_averaged_metrics.csv")
    log(f"[{task}] extra analyses -> {od}")


def main():
    df = pd.read_csv(out_dir("cohort") / "cohort_all.csv")
    df = df[df.eligible]
    t0 = time.time()
    log(f"Perturbed-mask descriptors for {len(df)} patients ({VARIANTS})")
    n_workers = int(_os.environ.get("DOLCHID_EXTRACT_JOBS", "3"))
    res = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(extract_patient)(r) for r in df[["patient_id", "cbct_path", "label_path"]].to_dict("records"))
    log(f"extraction done ({time.time() - t0:.0f}s)")
    for task in ["ALL4", "AME_vs_DC"]:
        analyse(task)
    log("Step 10 complete.")


if __name__ == "__main__":
    main()

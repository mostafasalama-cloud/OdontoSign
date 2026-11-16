"""
STEP 4: Statistical analysis.

1. Full-pipeline permutation tests (pre-specified representations).
   Labels are permuted at the PATIENT level, the stratified 5-fold split is
   regenerated, and the complete pipeline (standardisation, inner-CV C or
   attention-MIL training) is refitted for every permutation. The observed
   statistic is produced by exactly the same function with the true labels.
   One-sided, p = (1 + #{null >= observed}) / (N + 1)  (Phipson & Smyth 2010).

2. Bootstrap confidence intervals (repeat-0 out-of-fold predictions).
   Stratified patient resampling (class counts preserved, so no one-class
   samples), percentile intervals. These intervals are CONDITIONAL on the
   fitted out-of-fold predictions; variability from refitting is reported
   separately as the SD over CV repeats (step 3).

3. Pre-specified paired comparisons (same patients, same splits):
   * corrected resampled t-test (Nadeau & Bengio 2003) on per-fold
     differences over all repeats x folds -- primary test, two-sided;
   * paired stratified bootstrap CI of the difference (repeat 0);
   * exact two-sided McNemar test on hard predictions (per repeat; median p).
   Holm correction across the comparison family.

    python step4_statistics.py                 # all
    python step4_statistics.py --skip-perm     # skip permutation tests
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import config as CFG
from common import fast_auc, fast_ba, load_cohort, log, task_dir
from models import BASE_MODS, assemble, describe, run_cv


def comparison_family():
    fam = []
    for m in BASE_MODS:
        a = f"{m}|slic16_masked|attn"
        fam += [(a, f"{m}|whole|lr", "partition + attention vs whole-image"),
                (a, f"{m}|slic16_masked|mean", "attention vs mean pooling"),
                (a, f"{m}|slic16_masked|max", "attention vs max pooling"),
                (a, f"{m}|slic16_masked|concat16", "attention vs region concatenation"),
                (a, f"{m}|grid16_masked|attn", "SLIC vs grid regions"),
                (a, f"{m}|rand16_masked|attn", "SLIC vs random regions"),
                (f"{m}|slic16_crop|attn", a, "crop vs masked region input")]
    fam += [("MM_L|slic16_masked|attn", "HIST|slic16_masked|attn", "adding CBCT (attention)"),
            ("MM_L|whole|lr", "HIST|whole|lr", "adding CBCT (whole-image)"),
            ("MM_L|slic16_masked|attn", "MM_L|whole|lr", "multimodal attention vs whole-image"),
            ("CBCT_L|slic16_masked|attn", "CBCT_U|slic16_masked|attn", "lesion ROI vs uniform slices"),
            ("CBCT_L|whole|lr", "CBCT_U|whole|lr", "lesion ROI vs uniform slices (whole)")]
    return fam


# ----------------------------------------------------------------------------- permutation
def perm_once(rep, data, y, n_classes, i, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import config as CFG
    from common import fast_auc, fast_ba
    from models import run_cv
    rng = np.random.default_rng(CFG.SEED + 7919 * (i + 1))
    yp = y if i < 0 else rng.permutation(y)
    splits = list(StratifiedKFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
                  .split(np.zeros(len(yp)), yp))
    prob, _, _ = run_cv(rep, data, yp, splits, n_classes, repeat=0)
    return i, fast_ba(yp, prob.argmax(1), n_classes), fast_auc(yp, prob, n_classes)


def permutation_tests(y, pids, n_classes, out):
    rows, nulls = [], []
    for rep in CFG.PERMUTATION_REPS:
        t0 = time.time()
        data = assemble(rep, pids)
        res = Parallel(n_jobs=CFG.N_JOBS)(delayed(perm_once)(rep, data, y, n_classes, i)
                                          for i in range(-1, CFG.N_PERMUTATIONS))
        res = sorted(res)
        _, ba0, auc0 = res[0]
        nb = np.array([r[1] for r in res[1:]]); na = np.array([r[2] for r in res[1:]])
        N = len(nb)
        rows.append(dict(representation=rep, description=describe(rep), n_permutations=N,
                         seed=CFG.SEED, test="one-sided (null >= observed)",
                         observed_ba=ba0, null_ba_mean=nb.mean(), null_ba_sd=nb.std(),
                         p_ba=(1 + (nb >= ba0).sum()) / (N + 1),
                         observed_auc=auc0, null_auc_mean=np.nanmean(na), null_auc_sd=np.nanstd(na),
                         p_auc=(1 + (na >= auc0).sum()) / (N + 1),
                         p_floor=1 / (N + 1)))
        nulls += [dict(representation=rep, permutation=k, ba=b, auc=a)
                  for k, (b, a) in enumerate(zip(nb, na))]
        log(f"permutation {rep}: BA={ba0:.3f} p={rows[-1]['p_ba']:.4f} ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(out / "permutation_tests.csv", index=False)
    pd.DataFrame(nulls).to_csv(out / "permutation_null_distributions.csv.gz", index=False)


# ----------------------------------------------------------------------------- bootstrap
def strat_boot_indices(y, rng):
    return np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                           for c in np.unique(y)])


def bootstrap_cis(oof0, y, n_classes, out):
    rng = np.random.default_rng(CFG.SEED)
    idx = [strat_boot_indices(y, rng) for _ in range(CFG.N_BOOTSTRAP)]
    rows = []
    for rep, g in oof0.groupby("representation", sort=False):
        prob = g[[f"prob_{c}" for c in range(n_classes)]].to_numpy()
        pred = prob.argmax(1)
        ba = np.array([fast_ba(y[i], pred[i], n_classes) for i in idx])
        au = np.array([fast_auc(y[i], prob[i], n_classes) for i in idx])
        row = dict(representation=rep, description=describe(rep),
                   ba=fast_ba(y, pred, n_classes), ba_ci_low=np.percentile(ba, 2.5),
                   ba_ci_high=np.percentile(ba, 97.5),
                   auc=fast_auc(y, prob, n_classes), auc_ci_low=np.nanpercentile(au, 2.5),
                   auc_ci_high=np.nanpercentile(au, 97.5))
        if n_classes == 2:
            sens = np.array([(pred[i][y[i] == 1] == 1).mean() for i in idx])
            spec = np.array([(pred[i][y[i] == 0] == 0).mean() for i in idx])
            row.update(sensitivity=(pred[y == 1] == 1).mean(), sens_ci_low=np.percentile(sens, 2.5),
                       sens_ci_high=np.percentile(sens, 97.5),
                       specificity=(pred[y == 0] == 0).mean(), spec_ci_low=np.percentile(spec, 2.5),
                       spec_ci_high=np.percentile(spec, 97.5))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["method"] = f"stratified percentile bootstrap, B={CFG.N_BOOTSTRAP}, seed={CFG.SEED}, " \
                   "conditional on repeat-0 out-of-fold predictions"
    df.to_csv(out / "bootstrap_ci.csv", index=False)
    return idx


# ----------------------------------------------------------------------------- comparisons
def corrected_ttest(d, n_train, n_test):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    J = len(d)
    if J < 2:
        return (np.mean(d) if J else np.nan), np.nan, np.nan
    if np.var(d, ddof=1) == 0:  # identical fold results: no evidence either way
        return d.mean(), np.nan, (1.0 if d.mean() == 0 else np.nan)
    var = (1 / J + n_test / n_train) * np.var(d, ddof=1)
    t = d.mean() / np.sqrt(var)
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)


def comparisons(oof, folds, y, n_classes, boot_idx, out):
    rows = []
    reps_ok = set(oof.representation)
    n_train = folds.n_train.mean(); n_test = folds.n_test.mean()
    for a, b, question in comparison_family():
        if a not in reps_ok or b not in reps_ok:
            continue
        fa = folds[folds.representation == a].sort_values(["repeat", "fold"])
        fb = folds[folds.representation == b].sort_values(["repeat", "fold"])
        dba, t_ba, p_ba = corrected_ttest(fa.balanced_accuracy.values - fb.balanced_accuracy.values,
                                          n_train, n_test)
        dau, t_au, p_au = corrected_ttest(fa.roc_auc.values - fb.roc_auc.values, n_train, n_test)

        mc_p, b10, b01 = [], None, None
        for r in sorted(oof.repeat.unique()):
            pa = oof[(oof.representation == a) & (oof.repeat == r)].pred.values
            pb = oof[(oof.representation == b) & (oof.repeat == r)].pred.values
            ca, cb = pa == y, pb == y
            tab = [[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]]
            mc_p.append(mcnemar(tab, exact=True).pvalue)
            if r == 0:
                b10, b01 = tab[0][1], tab[1][0]

        g0a = oof[(oof.representation == a) & (oof.repeat == 0)]
        g0b = oof[(oof.representation == b) & (oof.repeat == 0)]
        pa = g0a[[f"prob_{c}" for c in range(n_classes)]].to_numpy()
        pb = g0b[[f"prob_{c}" for c in range(n_classes)]].to_numpy()
        dbb = [fast_ba(y[i], pa[i].argmax(1), n_classes) - fast_ba(y[i], pb[i].argmax(1), n_classes)
               for i in boot_idx]
        dab = [fast_auc(y[i], pa[i], n_classes) - fast_auc(y[i], pb[i], n_classes) for i in boot_idx]
        rows.append(dict(model_a=a, model_b=b, question=question,
                         delta_ba_mean_folds=dba, t_ba=t_ba, p_ba_corrected_t=p_ba,
                         delta_auc_mean_folds=dau, t_auc=t_au, p_auc_corrected_t=p_au,
                         delta_ba_rep0=fast_ba(y, pa.argmax(1), n_classes) - fast_ba(y, pb.argmax(1), n_classes),
                         delta_ba_boot_ci_low=np.percentile(dbb, 2.5),
                         delta_ba_boot_ci_high=np.percentile(dbb, 97.5),
                         delta_auc_rep0=fast_auc(y, pa, n_classes) - fast_auc(y, pb, n_classes),
                         delta_auc_boot_ci_low=np.nanpercentile(dab, 2.5),
                         delta_auc_boot_ci_high=np.nanpercentile(dab, 97.5),
                         mcnemar_only_a_correct_rep0=b10, mcnemar_only_b_correct_rep0=b01,
                         mcnemar_p_rep0=mc_p[0], mcnemar_p_median=float(np.median(mc_p)),
                         mcnemar_frac_repeats_p05=float(np.mean(np.array(mc_p) < 0.05))))
    df = pd.DataFrame(rows)
    if len(df):
        for col in ("p_ba_corrected_t", "p_auc_corrected_t"):
            ok = df[col].notna()
            df.loc[ok, col.replace("p_", "p_holm_")] = multipletests(df.loc[ok, col], method="holm")[1]
    df.to_csv(out / "paired_comparisons.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-perm", action="store_true")
    args = ap.parse_args()

    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    n_classes = len(CFG.TASK_CLASSES[CFG.TASK])
    cv = task_dir("cv"); out = task_dir("stats")
    oof = pd.read_csv(cv / "oof_predictions.csv.gz")
    folds = pd.read_csv(cv / "fold_metrics.csv")
    oof = oof.sort_values(["representation", "repeat"], kind="stable").reset_index(drop=True)
    oof0 = oof[oof.repeat == 0]
    for rep, g in oof0.groupby("representation"):
        assert list(g.patient_id) == pids, f"patient order mismatch in {rep}"

    log("Bootstrap confidence intervals")
    boot_idx = bootstrap_cis(oof0, y, n_classes, out)
    log("Paired comparisons")
    comp = comparisons(oof, folds, y, n_classes, boot_idx, out)
    if len(comp):
        print(comp[["model_a", "model_b", "delta_ba_mean_folds", "p_ba_corrected_t",
                    "p_holm_ba_corrected_t", "mcnemar_p_median"]].round(4).to_string(index=False))
    if not args.skip_perm:
        log(f"Permutation tests ({CFG.N_PERMUTATIONS} permutations each, full refit)")
        permutation_tests(y, pids, n_classes, out)
    log(f"Step 4 complete -> {out}")


if __name__ == "__main__":
    main()

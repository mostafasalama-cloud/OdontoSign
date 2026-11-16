"""
STEP 3: Leakage-free repeated stratified cross-validation of every representation.

Every label-dependent component (feature standardisation, logistic-regression C,
attention network, classifier head) is fitted inside the training fold only.
All representations share exactly the same splits, so comparisons are paired.

Outputs (output/results/<TASK>/cv/):
    oof_predictions.csv.gz   per representation x repeat x patient probabilities
    fold_metrics.csv         per fold (fold-averaged metrics are reported separately)
    repeat_metrics.csv       metrics of pooled out-of-fold predictions, per repeat
    summary.csv              mean, SD and 2.5-97.5 percentile over repeats
    fold_class_counts.csv    patients per class in every test fold
    attention_weights.csv.gz held-out attention weights (repeat 0)
Resumable: each representation is cached in cv/cache/.

    python step3_evaluate.py               # everything in the registry
    python step3_evaluate.py --only HIST   # representations whose name contains HIST
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

import config as CFG
from common import compute_metrics, load_cohort, log, task_dir
from models import assemble, describe, parse_rep, representation_registry, run_cv


def make_splits(y):
    rskf = RepeatedStratifiedKFold(n_splits=CFG.N_FOLDS, n_repeats=CFG.N_REPEATS,
                                   random_state=CFG.SEED)
    all_splits = list(rskf.split(np.zeros(len(y)), y))
    return [all_splits[r * CFG.N_FOLDS:(r + 1) * CFG.N_FOLDS] for r in range(CFG.N_REPEATS)]


def run_hash(pids):
    key = dict(p=list(pids), R=CFG.N_REPEATS, K=CFG.N_FOLDS, seed=CFG.SEED, mil=CFG.MIL,
               cs=CFG.LR_CS, task=CFG.TASK)
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def one_repeat(rep, data, y, splits, n_classes, r, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from models import run_cv
    prob, rows, att = run_cv(rep, data, y, splits, n_classes, repeat=r, keep_attention=(r == 0))
    return r, prob, rows, att


def ensure_current_features(pids):
    """Re-run step 2 if any feature file was made with different settings."""
    import subprocess
    import sys
    from step2_extract_features import already_done
    for attempt in range(2):
        stale = [p for p in pids for m in CFG.CBCT_MODES + ["HIST"] if not already_done(m, p)]
        if not stale:
            return
        log(f"{len(stale)} feature files missing or made with old settings -> re-running step 2")
        subprocess.run([sys.executable, str(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                                          "step2_extract_features.py"))], check=True)
    sys.exit("Features still not current after re-running step 2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on representation names")
    args = ap.parse_args()

    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    ensure_current_features(pids)
    classes = CFG.TASK_CLASSES[CFG.TASK]
    n_classes = len(classes)
    log(f"Task {CFG.TASK}: {len(y)} patients | " +
        ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(classes)))
    splits = make_splits(y)
    out = task_dir("cv")
    cache = task_dir("cv", "cache")
    h = run_hash(pids)

    rows = []
    for r, sp in enumerate(splits):
        for f, (tr, te) in enumerate(sp):
            rows.append(dict(repeat=r, fold=f, n_train=len(tr), n_test=len(te),
                             **{f"test_{c}": int((y[te] == i).sum()) for i, c in enumerate(classes)},
                             **{f"train_{c}": int((y[tr] == i).sum()) for i, c in enumerate(classes)}))
    pd.DataFrame(rows).to_csv(out / "fold_class_counts.csv", index=False)

    reps = [r for r in representation_registry() if args.only in r]
    log(f"Representations to evaluate: {len(reps)} x {CFG.N_REPEATS} repeats x {CFG.N_FOLDS} folds")
    for i, rep in enumerate(reps, 1):
        cfile = cache / (rep.replace("|", "__") + ".pkl")
        if cfile.exists():
            with open(cfile, "rb") as f:
                if pickle.load(f).get("hash") == h:
                    continue
        t0 = time.time()
        data = assemble(rep, pids)
        res = Parallel(n_jobs=CFG.N_JOBS)(
            delayed(one_repeat)(rep, data, y, splits[r], n_classes, r)
            for r in range(CFG.N_REPEATS))
        with open(cfile, "wb") as f:
            pickle.dump(dict(hash=h, rep=rep, results=res), f)
        ba = np.mean([compute_metrics(y, p, n_classes)["balanced_accuracy"] for _, p, _, _ in res])
        log(f"[{i}/{len(reps)}] {rep:40s} BA={ba:.3f}  ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- collect
    oof, folds, rmet, att_rows = [], [], [], []
    for rep in representation_registry():
        cfile = cache / (rep.replace("|", "__") + ".pkl")
        if not cfile.exists():
            continue
        with open(cfile, "rb") as f:
            obj = pickle.load(f)
        if obj["hash"] != h:
            continue
        mods = parse_rep(rep)[0]
        for r, prob, frows, att in obj["results"]:
            for fr in frows:
                folds.append(dict(representation=rep, **fr))
            rmet.append(dict(representation=rep, repeat=r, **compute_metrics(y, prob, n_classes)))
            df = pd.DataFrame(prob, columns=[f"prob_{c}" for c in range(n_classes)])
            df.insert(0, "patient_id", pids); df.insert(0, "repeat", r)
            df.insert(0, "representation", rep)
            df["label"] = y; df["pred"] = prob.argmax(1)
            df["fold"] = -1
            for f, (_, te) in enumerate(splits[r]):
                df.loc[te, "fold"] = f
            oof.append(df)
            for te_i, atts in att:
                for mod, a in zip(mods, atts):
                    for j, pi in enumerate(te_i):
                        for k, w in enumerate(a[j]):
                            if w > 0:
                                att_rows.append((rep, mod, pids[pi], k, float(w)))

    pd.concat(oof).to_csv(out / "oof_predictions.csv.gz", index=False)
    folds = pd.DataFrame(folds); folds.to_csv(out / "fold_metrics.csv", index=False)
    rmet = pd.DataFrame(rmet); rmet.to_csv(out / "repeat_metrics.csv", index=False)
    pd.DataFrame(att_rows, columns=["representation", "modality", "patient_id", "region", "weight"])\
        .to_csv(out / "attention_weights.csv.gz", index=False)

    metrics = [c for c in rmet.columns if c not in ("representation", "repeat")]
    summ = []
    for rep, g in rmet.groupby("representation", sort=False):
        row = dict(representation=rep, description=describe(rep))
        for m in metrics:
            v = g[m].to_numpy(float)
            row[f"{m}_mean"] = np.nanmean(v); row[f"{m}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else 0
            row[f"{m}_p2.5"] = np.nanpercentile(v, 2.5); row[f"{m}_p97.5"] = np.nanpercentile(v, 97.5)
        fr = folds[(folds.representation == rep) & (folds.repeat == 0)]
        for m in ("balanced_accuracy", "roc_auc", "accuracy"):
            row[f"{m}_foldavg_rep0_mean"] = fr[m].mean()
            row[f"{m}_foldavg_rep0_sd"] = fr[m].std()
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values("balanced_accuracy_mean", ascending=False)
    summ.to_csv(out / "summary.csv", index=False)
    print("\nPooled out-of-fold metrics, mean (SD) over repeats:")
    for _, r in summ.iterrows():
        print(f"  {r.representation:40s} BA {r.balanced_accuracy_mean:.3f} ({r.balanced_accuracy_sd:.3f})"
              f"  AUC {r.roc_auc_mean:.3f} ({r.roc_auc_sd:.3f})")
    log(f"Step 3 complete -> {out}")


if __name__ == "__main__":
    main()

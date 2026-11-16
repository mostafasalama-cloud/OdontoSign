"""
STEP 7 (CBCT-centred study): repeated stratified CV of every pre-specified
representation (same splits as step 3), plus the 3D CNN results of step 8 and
post-hoc late fusions. Outputs in results/<TASK>/cbct_study/.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as CFG
from cbct_study import GROUP, LATE, REGISTRY, assemble, label, run_cv, study_dir
from common import compute_metrics, load_cohort, log
from step3_evaluate import make_splits


def run_hash(pids):
    return hashlib.md5(json.dumps(dict(p=list(pids), R=CFG.N_REPEATS, K=CFG.N_FOLDS, s=CFG.SEED,
                                       t=CFG.TASK, v=2)).encode()).hexdigest()[:10]


def one_repeat(name, data, y, splits, n_classes, r, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    from cbct_study import run_cv
    prob, rows = run_cv(name, data, y, splits, n_classes, repeat=r)
    return r, prob, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    classes = CFG.TASK_CLASSES[CFG.TASK]; n_classes = len(classes)
    splits = make_splits(y)
    cache = study_dir("cache"); out = study_dir()
    h = run_hash(pids)
    log(f"CBCT study, task {CFG.TASK}: {len(y)} patients")

    for name, spec in REGISTRY.items():
        if spec["learner"] == "cnn3d" or args.only not in name:
            continue
        cf = cache / f"{name}.pkl"
        if cf.exists():
            with open(cf, "rb") as f:
                if pickle.load(f).get("hash") == h:
                    continue
        t0 = time.time()
        data = assemble(name, pids)
        res = Parallel(n_jobs=CFG.N_JOBS)(delayed(one_repeat)(name, data, y, splits[r], n_classes, r)
                                          for r in range(CFG.N_REPEATS))
        with open(cf, "wb") as f:
            pickle.dump(dict(hash=h, results=res), f)
        ba = np.mean([compute_metrics(y, p, n_classes)["balanced_accuracy"] for _, p, _ in res])
        log(f"{name:28s} BA={ba:.3f} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------- collect
    results = {}
    for name in REGISTRY:
        cf = cache / f"{name}.pkl"
        if cf.exists():
            with open(cf, "rb") as f:
                obj = pickle.load(f)
            if obj["hash"] == h:
                results[name] = {r: (p, rows) for r, p, rows in obj["results"]}
    for lname, comps in LATE.items():
        if all(c in results for c in comps):
            common_r = sorted(set.intersection(*[set(results[c]) for c in comps]))
            results[lname] = {}
            for r in common_r:
                p = np.mean([results[c][r][0] for c in comps], axis=0)
                rows = []
                for f, (tr, te) in enumerate(splits[r]):
                    rows.append(dict(repeat=r, fold=f, n_train=len(tr), n_test=len(te),
                                     **compute_metrics(y[te], p[te], n_classes)))
                results[lname][r] = (p, rows)

    oof, folds, rmet = [], [], []
    for name, reps in results.items():
        for r, (p, rows) in sorted(reps.items()):
            folds += [dict(representation=name, **x) for x in rows]
            rmet.append(dict(representation=name, repeat=r, **compute_metrics(y, p, n_classes)))
            d = pd.DataFrame(p, columns=[f"prob_{c}" for c in range(n_classes)])
            d.insert(0, "patient_id", pids); d.insert(0, "repeat", r); d.insert(0, "representation", name)
            d["label"] = y; d["pred"] = p.argmax(1)
            oof.append(d)
    pd.concat(oof).to_csv(out / "oof_predictions.csv.gz", index=False)
    folds = pd.DataFrame(folds); folds.to_csv(out / "fold_metrics.csv", index=False)
    rmet = pd.DataFrame(rmet); rmet.to_csv(out / "repeat_metrics.csv", index=False)
    metrics = [c for c in rmet.columns if c not in ("representation", "repeat")]
    summ = []
    for name, g in rmet.groupby("representation", sort=False):
        row = dict(representation=name, group=GROUP.get(name, ""), description=label(name),
                   n_repeats=len(g))
        for m in metrics:
            v = g[m].to_numpy(float)
            row[f"{m}_mean"] = np.nanmean(v); row[f"{m}_sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else 0
        summ.append(row)
    summ = pd.DataFrame(summ).sort_values(["group", "balanced_accuracy_mean"], ascending=[True, False])
    summ.to_csv(out / "summary.csv", index=False)
    print("\nPooled OOF metrics, mean (SD) over repeats:")
    for _, r in summ.iterrows():
        print(f"  [{r.group:4s}] {r.representation:28s} BA {r.balanced_accuracy_mean:.3f} ({r.balanced_accuracy_sd:.3f})"
              f"  AUC {r.roc_auc_mean:.3f}  acc {r.accuracy_mean:.3f}  F1 {r.f1_macro_mean:.3f}  (R={r.n_repeats})")
    log(f"Step 7 complete -> {out}")


if __name__ == "__main__":
    main()

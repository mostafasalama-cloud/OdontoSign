"""
STEP 11 (exploratory, added after step 10 showed sensitivity to mask errors).

Perturbation-augmented training of the descriptor model. Inside each training
fold the logistic-regression model is trained on the descriptors of the
training patients computed from the expert mask AND from perturbed masks
(erosion/dilation by 1 and 2 mm), all with the patient's label; it is then
applied to the held-out patients under each mask condition.

  aug_all   : trained on expert + erode1 + erode2 + dilate1 + dilate2
  aug_lopo  : as aug_all but leaving out the perturbation used at test time
              (tests generalisation to an unseen type of mask error)
  bbox is never used for training here (out-of-distribution test).

Also: robustness of the expert-trained model when one descriptor family is
removed (which families make the model fragile).
Outputs: results/<TASK>/cbct_study/extra/mask_augmentation.csv, family_robustness.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config as C
from common import compute_metrics, load_cohort, log
from step10_cbct_extra import GROUPS, VARIANTS, fit_lr, rad_matrix

PERTURB = ["erode1", "erode2", "dilate1", "dilate2"]


def analyse(task):
    C.TASK = task
    from cbct_study import study_dir
    from step3_evaluate import make_splits
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    nc = len(C.TASK_CLASSES[task])
    splits = make_splits(y)
    X0, names = rad_matrix(pids)
    XV = {"expert": X0, **{v: rad_matrix(pids, v)[0] for v in VARIANTS}}
    od = study_dir("extra")

    def run(r):
        rows = []
        for f, (tr, te) in enumerate(splits[r]):
            seed = C.SEED + 1000 * r + f
            train_sets = ["expert"] + PERTURB
            clf_all = fit_lr(np.vstack([XV[v][tr] for v in train_sets]), np.tile(y[tr], len(train_sets)), seed)
            for test in ["expert"] + VARIANTS:
                rows.append(dict(repeat=r, fold=f, train="aug_all", test=test,
                                 **compute_metrics(y[te], clf_all.predict_proba(XV[test][te]), nc)))
                if test in PERTURB:
                    ts = ["expert"] + [v for v in PERTURB if v != test]
                    clf = fit_lr(np.vstack([XV[v][tr] for v in ts]), np.tile(y[tr], len(ts)), seed)
                    rows.append(dict(repeat=r, fold=f, train="aug_lopo", test=test,
                                     **compute_metrics(y[te], clf.predict_proba(XV[test][te]), nc)))
            # family ablation of the expert-trained model
            for g, cols in [("none", [])] + list(GROUPS.items()):
                keep = [i for i, n in enumerate(names) if n not in cols]
                clf = fit_lr(X0[tr][:, keep], y[tr], seed)
                for test in ["expert", "erode1", "dilate1"]:
                    rows.append(dict(repeat=r, fold=f, train=f"expert_minus_{g}", test=test,
                                     **compute_metrics(y[te], clf.predict_proba(XV[test][te][:, keep]), nc)))
        return rows

    res = pd.DataFrame(sum(Parallel(n_jobs=C.N_JOBS)(delayed(run)(r) for r in range(C.N_REPEATS)), []))
    agg = (res.groupby(["train", "test"])[["balanced_accuracy", "accuracy", "f1_macro", "roc_auc"]]
           .agg(["mean", "std"]).round(3))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    aug = agg[agg.train.str.startswith("aug")]
    fam = agg[agg.train.str.startswith("expert_minus")]
    aug.to_csv(od / "mask_augmentation.csv", index=False)
    fam.to_csv(od / "family_robustness.csv", index=False)
    log(f"[{task}] augmentation:\n{aug.to_string(index=False)}")
    log(f"[{task}] family robustness (BA):\n" +
        fam.pivot(index="train", columns="test", values="balanced_accuracy_mean").round(3).to_string())


def main():
    for task in ["ALL4", "AME_vs_DC"]:
        analyse(task)
    log("Step 11 complete.")


if __name__ == "__main__":
    main()

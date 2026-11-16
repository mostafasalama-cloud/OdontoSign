"""
STEP 9 (CBCT-centred study): statistics, tables and figures.

* Bootstrap 95% CIs (stratified, repeat 0, conditional on OOF predictions)
* Pre-specified paired comparisons: Nadeau-Bengio corrected resampled t-test
  over matching repeat x fold pairs (two-sided), Holm-adjusted; exact McNemar
* Full-pipeline permutation tests (patient-level label permutation, complete
  refit incl. inner CV / distillation teacher) for pre-specified models
* Comparison with the published DOLCHID benchmark (Sci Data 2026, Tables 4-5)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

import config as CFG
from cbct_study import GROUP, PRIMARY, assemble, label, study_dir
from common import fast_auc, fast_ba, load_cohort, log
from step4_statistics import corrected_ttest, strat_boot_indices

COMPARISONS = [
    (PRIMARY, "C2D_lesion", "3D descriptors + foundation 2.5D vs 2D lesion summary"),
    (PRIMARY, "RAD", "adding 2.5D deep features to descriptors"),
    (PRIMARY, "C25_dinob", "adding descriptors to 2.5D deep features"),
    ("C25_dinob", "C25_r18", "DINOv2-B vs ImageNet ResNet-18 (2.5D)"),
    ("C25_dinob_attn", "C25_dinob", "attention-MIL vs mean over views"),
    ("RAD+C25_dinob_KD", "RAD+C25_dinob_MLP", "histology distillation vs none (same MLP)"),
    ("CNN3D", PRIMARY, "3D CNN vs primary"),
    ("LATE(RAD+C25_dinob,CNN3D)", PRIMARY, "adding 3D CNN by late fusion"),
    ("HT_dinob", "H_whole_r18", "histology: DINOv2-B ROI tiles vs ResNet-18 whole image"),
    ("MM_RAD+C25+HT", "HT_dinob", "adding CBCT to histology"),
    ("HT_dinob", PRIMARY, "histology vs CBCT (modality gap)"),
]
PERM_MODELS = [PRIMARY, "RAD+C25_dinob_KD", "RAD", "HT_dinob"]

# Published DOLCHID benchmark, 4-class, 5-fold CV, mean (SD) in % (Sci Data 2026, Tables 4-5)
BENCHMARK = pd.DataFrame([
    ("CBCT", "ResNet34 (3D)", 54.6, 52.3, 77.1), ("CBCT", "ResNet50 (3D)", 51.9, 48.2, 74.3),
    ("CBCT", "AttResNet34 (3D)", 58.0, 56.5, 80.1), ("CBCT", "M3T", 53.0, 51.3, 72.7),
    ("H&E", "CLAM, ResNet50 enc.", 78.2, 77.2, 95.0), ("H&E", "CLAM, UNI enc.", 95.0, 94.7, 98.9),
    ("H&E", "CLAM, CONCH enc.", 90.1, 89.6, 98.2),
    ("CBCT+H&E", "CLIP fusion", 59.2, 58.3, 75.4), ("CBCT+H&E", "GFF fusion", 60.7, 60.1, 72.3),
], columns=["modality", "model", "accuracy_%", "F1_%", "AUC_%"])


def perm_once(name, data, y, n_classes, i, here=_os.path.dirname(_os.path.abspath(__file__))):
    import sys
    if here not in sys.path:
        sys.path.insert(0, here)
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    import config as CFG
    from cbct_study import run_cv
    from common import fast_auc, fast_ba
    rng = np.random.default_rng(CFG.SEED + 7919 * (i + 1))
    yp = y if i < 0 else rng.permutation(y)
    sp = list(StratifiedKFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED).split(np.zeros(len(yp)), yp))
    p, _ = run_cv(name, data, yp, sp, n_classes)
    return i, fast_ba(yp, p.argmax(1), n_classes), fast_auc(yp, p, n_classes)


def fmt(m, s):
    return "–" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"


def md(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-perm", action="store_true")
    args = ap.parse_args()
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    classes = CFG.TASK_CLASSES[CFG.TASK]; n_classes = len(classes)
    d = study_dir()
    summ = pd.read_csv(d / "summary.csv").set_index("representation")
    folds = pd.read_csv(d / "fold_metrics.csv")
    oof = pd.read_csv(d / "oof_predictions.csv.gz")
    oof = oof.sort_values(["representation", "repeat"], kind="stable")
    have = set(summ.index)
    pc = [f"prob_{c}" for c in range(n_classes)]
    out_md = [f"# CBCT-centred study — task {CFG.TASK} ({len(y)} patients: " +
              ", ".join(f"{c} {int((y == i).sum())}" for i, c in enumerate(classes)) + ")\n",
              f"Primary CBCT model (pre-specified): **{PRIMARY}** — {label(PRIMARY)}\n"]

    # ------------------------------------------------------------ main table
    rows = []
    for name, s in summ.iterrows():
        r = dict(group=s.group, model=name, description=s.description, repeats=int(s.n_repeats),
                 BA=fmt(s.balanced_accuracy_mean, s.balanced_accuracy_sd),
                 accuracy=fmt(s.accuracy_mean, s.accuracy_sd),
                 F1_macro=fmt(s.f1_macro_mean, s.f1_macro_sd), AUC=fmt(s.roc_auc_mean, s.roc_auc_sd))
        if n_classes == 2:
            r.update(sensitivity=fmt(s.sensitivity_mean, s.sensitivity_sd),
                     specificity=fmt(s.specificity_mean, s.specificity_sd))
        else:
            for i, c in enumerate(classes):
                r[f"recall_{c}"] = fmt(s[f"recall_class{i}_mean"], s[f"recall_class{i}_sd"])
        rows.append(r)
    main_t = pd.DataFrame(rows)
    main_t.to_csv(d / "table_results.csv", index=False)
    out_md += ["\n## Results (pooled out-of-fold, mean ± SD over CV repeats)\n", md(main_t)]

    # ------------------------------------------------------------ bootstrap
    rng = np.random.default_rng(CFG.SEED)
    bidx = [strat_boot_indices(y, rng) for _ in range(CFG.N_BOOTSTRAP)]
    brows = []
    for name, g in oof[oof.repeat == 0].groupby("representation", sort=False):
        p = g[pc].to_numpy(); pr = p.argmax(1)
        ba = [fast_ba(y[i], pr[i], n_classes) for i in bidx]
        au = [fast_auc(y[i], p[i], n_classes) for i in bidx]
        acc = [(pr[i] == y[i]).mean() for i in bidx]
        brows.append(dict(model=name, BA=f"{fast_ba(y, pr, n_classes):.3f} [{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]",
                          accuracy=f"{(pr == y).mean():.3f} [{np.percentile(acc, 2.5):.3f}, {np.percentile(acc, 97.5):.3f}]",
                          AUC=f"{fast_auc(y, p, n_classes):.3f} [{np.nanpercentile(au, 2.5):.3f}, {np.nanpercentile(au, 97.5):.3f}]"))
    bt = pd.DataFrame(brows); bt.to_csv(d / "table_bootstrap.csv", index=False)
    out_md += [f"\n## 95% bootstrap CIs (repeat 0; stratified percentile bootstrap, B={CFG.N_BOOTSTRAP}, "
               "conditional on OOF predictions)\n", md(bt)]

    # ------------------------------------------------------------ comparisons
    crow = []
    for a, b, q in COMPARISONS:
        if a not in have or b not in have:
            continue
        fa = folds[folds.representation == a].set_index(["repeat", "fold"])
        fb = folds[folds.representation == b].set_index(["repeat", "fold"])
        j = fa.index.intersection(fb.index)
        res = dict(question=q, model_a=a, model_b=b, n_pairs=len(j))
        for m in ("balanced_accuracy", "roc_auc", "accuracy"):
            dm, t, p = corrected_ttest(fa.loc[j, m].values - fb.loc[j, m].values,
                                       fa.loc[j, "n_train"].mean(), fa.loc[j, "n_test"].mean())
            res[f"delta_{m}"] = dm; res[f"p_{m}"] = p
        mc = []
        for r in sorted(set(oof[oof.representation == a].repeat) & set(oof[oof.representation == b].repeat)):
            ca = oof[(oof.representation == a) & (oof.repeat == r)].pred.values == y
            cb = oof[(oof.representation == b) & (oof.repeat == r)].pred.values == y
            mc.append(mcnemar([[(ca & cb).sum(), (ca & ~cb).sum()], [(~ca & cb).sum(), (~ca & ~cb).sum()]],
                              exact=True).pvalue)
        res["mcnemar_p_median"] = float(np.median(mc))
        crow.append(res)
    ct = pd.DataFrame(crow)
    if len(ct):
        for m in ("balanced_accuracy", "roc_auc"):
            ok = ct[f"p_{m}"].notna()
            ct.loc[ok, f"p_holm_{m}"] = multipletests(ct.loc[ok, f"p_{m}"], method="holm")[1]
        ct.to_csv(d / "table_comparisons.csv", index=False)
        show = ct[["question", "model_a", "model_b", "n_pairs", "delta_balanced_accuracy", "p_balanced_accuracy",
                   "p_holm_balanced_accuracy", "delta_roc_auc", "p_holm_roc_auc", "mcnemar_p_median"]]
        out_md += ["\n## Pre-specified paired comparisons (corrected resampled t-test, Holm-adjusted)\n", md(show)]

    # ------------------------------------------------------------ confusion + per class
    g = oof[(oof.representation == PRIMARY) & (oof.repeat == 0)]
    if len(g):
        cm = confusion_matrix(g.label, g.pred, labels=list(range(n_classes)))
        cmt = pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes])
        cmt.to_csv(d / "table_confusion_primary.csv")
        out_md += ["\n## Confusion matrix, primary CBCT model (repeat 0)\n", md(cmt.reset_index())]
        fig, ax = plt.subplots(figsize=(4, 3.4))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(n_classes):
            for k in range(n_classes):
                ax.text(k, i, cm[i, k], ha="center", va="center", fontsize=9)
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes)
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Primary CBCT model")
        fig.tight_layout(); fig.savefig(d / "fig_confusion_primary.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------ benchmark
    if CFG.TASK == "ALL4" and not CFG.RESTRICT_TO_ORIGINAL_100:
        ours = []
        for name in [PRIMARY, "RAD+C25_dinob_KD", "CNN3D", "LATE(RAD+C25_dinob,CNN3D)", "C2D_lesion",
                     "HT_dinob", "MM_RAD+C25+HT"]:
            if name in have:
                s = summ.loc[name]
                ours.append(dict(modality={"CBCT": "CBCT", "HIST": "H&E", "MM": "CBCT+H&E"}[s.group],
                                 model=f"OURS: {name}", **{"accuracy_%": 100 * s.accuracy_mean,
                                                            "F1_%": 100 * s.f1_macro_mean,
                                                            "AUC_%": 100 * s.roc_auc_mean}))
        bench = pd.concat([BENCHMARK.assign(source="DOLCHID paper"),
                           pd.DataFrame(ours).assign(source="this study")]).round(1)
        bench.to_csv(d / "table_benchmark.csv", index=False)
        out_md += ["\n## Comparison with the published DOLCHID benchmark (4-class)\n",
                   "_Published values: 5-fold CV, mean of folds (Sci Data 2026, Tables 4–6). Ours: pooled OOF, mean "
                   "over 10 repeats of stratified 5-fold CV; F1 = macro. Protocols are similar but not identical._\n",
                   md(bench)]

    # ------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(summ) + 1))
    s2 = summ.sort_values("balanced_accuracy_mean")
    col = {"CBCT": "#3b7dd8", "HIST": "#c2571a", "MM": "#6b8e23"}
    ax.barh(range(len(s2)), s2.balanced_accuracy_mean, xerr=s2.balanced_accuracy_sd,
            color=[col.get(gp, "grey") for gp in s2.group])
    ax.set_yticks(range(len(s2))); ax.set_yticklabels(s2.index, fontsize=7)
    ax.axvline(1 / n_classes, ls=":", c="k"); ax.set_xlabel("balanced accuracy (mean ± SD over repeats)")
    fig.tight_layout(); fig.savefig(d / "fig_models.png", dpi=200); plt.close(fig)

    # ------------------------------------------------------------ permutations
    if not args.skip_perm:
        prow = []
        for name in PERM_MODELS:
            if name not in have:
                continue
            t0 = time.time()
            data = assemble(name, pids)
            res = sorted(Parallel(n_jobs=CFG.N_JOBS)(delayed(perm_once)(name, data, y, n_classes, i)
                                                     for i in range(-1, CFG.N_PERMUTATIONS)))
            _, ba0, au0 = res[0]
            nb = np.array([r[1] for r in res[1:]]); na = np.array([r[2] for r in res[1:]])
            N = len(nb)
            prow.append(dict(model=name, n_permutations=N, observed_ba=ba0, null_ba_mean=nb.mean(),
                             p_ba=(1 + (nb >= ba0).sum()) / (N + 1), observed_auc=au0,
                             null_auc_mean=np.nanmean(na), p_auc=(1 + (na >= au0).sum()) / (N + 1)))
            log(f"permutation {name}: BA={ba0:.3f} p={prow[-1]['p_ba']:.4f} ({time.time() - t0:.0f}s)")
        pt = pd.DataFrame(prow); pt.to_csv(d / "table_permutation.csv", index=False)
    if (d / "table_permutation.csv").exists():
        out_md += ["\n## Full-pipeline permutation tests (one-sided)\n", md(pd.read_csv(d / "table_permutation.csv"))]
    (d / "CBCT_STUDY_SUMMARY.md").write_text("\n".join(out_md), encoding="utf-8")
    log(f"Step 9 complete -> {d / 'CBCT_STUDY_SUMMARY.md'}")


if __name__ == "__main__":
    main()

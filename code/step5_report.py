"""
STEP 5: Paper-ready tables, figures and a results inventory.

Outputs (output/results/<TASK>/report/):
    RESULTS_SUMMARY.md      all tables in markdown, in manuscript order
    tables/*.csv            the same tables as CSV
    figures/*.png           ROC curves, ablation plot, permutation nulls,
                            partition examples, held-out attention maps
    results_inventory.csv   provenance of every table/figure (reviewer item 4)
    run_config.json         every setting + library versions
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from sklearn.metrics import confusion_matrix, roc_curve

import config as CFG
from common import (config_snapshot, environment_info, feature_path, load_cohort, log, out_dir,
                    save_json, task_dir)
from models import describe

PRIMARY_MODS = ["HIST", "CBCT_U", "CBCT_L", "MM_U", "MM_L"]
PRIMARY = [f"{m}|{k}" for m in PRIMARY_MODS for k in ("whole|lr", "slic16_masked|attn",
                                                       "slic16_masked|mean")]


def fmt(m, s):
    return "–" if pd.isna(m) else f"{m:.3f} ± {s:.3f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else
                                         (f"{v:.4f}" if isinstance(v, float) else str(v))
                                         for v in r) + " |")
    return "\n".join(lines)


def main():
    cohort = load_cohort()
    classes = CFG.TASK_CLASSES[CFG.TASK]
    n_classes = len(classes)
    y = cohort.label.to_numpy()
    cv, st = task_dir("cv"), task_dir("stats")
    rep_dir = task_dir("report"); tdir = task_dir("report", "tables"); fdir = task_dir("report", "figures")
    summ = pd.read_csv(cv / "summary.csv").set_index("representation")
    oof = pd.read_csv(cv / "oof_predictions.csv.gz")
    oof = oof.sort_values(["representation", "repeat"], kind="stable")
    fcc = pd.read_csv(cv / "fold_class_counts.csv")
    have = set(summ.index)
    md, inv = [f"# Results summary — task {CFG.TASK}\n"], []

    def add(name, title, df, source, note=""):
        df.to_csv(tdir / f"{name}.csv", index=False)
        md.append(f"\n## {title}\n")
        if note:
            md.append(f"_{note}_\n")
        md.append(md_table(df))
        inv.append(dict(item=name, title=title, source=source, note=note))

    # ---- cohort
    flow = pd.read_csv(out_dir("cohort") / "cohort_flow.csv")
    add("table_cohort_flow", "Case-selection flow (all DOLCHID classes)", flow,
        "step1 cohort_flow.csv", "Inclusion: CBCT image + CBCT lesion label + histopathology "
        "image present, readable and matched by patient ID.")
    counts = pd.DataFrame([dict(clinical_class=c, label=i, n=int((y == i).sum())) for i, c in enumerate(classes)])
    maj = counts.n.max() / counts.n.sum()
    add("table_task_cohort", "Analysed cohort", counts, "step1/common.load_cohort",
        f"Majority-class accuracy = {maj:.3f}; chance balanced accuracy = {1 / n_classes:.3f}.")
    add("table_fold_counts", "Test-fold class counts (repeat 0)",
        fcc[fcc.repeat == 0].drop(columns="repeat"), "step3 fold_class_counts.csv",
        "Each patient appears in exactly one test fold per repeat.")

    # ---- main results
    rows = []
    for rep in PRIMARY:
        if rep not in have:
            continue
        s = summ.loc[rep]
        r = dict(representation=describe(rep), BA=fmt(s.balanced_accuracy_mean, s.balanced_accuracy_sd),
                 AUC=fmt(s.roc_auc_mean, s.roc_auc_sd), accuracy=fmt(s.accuracy_mean, s.accuracy_sd))
        if n_classes == 2:
            r.update(sensitivity=fmt(s.sensitivity_mean, s.sensitivity_sd),
                     specificity=fmt(s.specificity_mean, s.specificity_sd),
                     F1_DC=fmt(s.f1_positive_mean, s.f1_positive_sd))
        else:
            r.update(F1_macro=fmt(s.f1_macro_mean, s.f1_macro_sd))
        r["BA fold-avg (rep 0)"] = fmt(s.balanced_accuracy_foldavg_rep0_mean,
                                       s.balanced_accuracy_foldavg_rep0_sd)
        rows.append(r)
    add("table_main_results", "Main results", pd.DataFrame(rows), "step3 summary.csv",
        f"Pooled out-of-fold metrics per repeat, mean ± SD over {CFG.N_REPEATS} repeats of "
        f"stratified {CFG.N_FOLDS}-fold CV. Last column: mean ± SD of per-fold BA in repeat 0 "
        "(fold-averaged, reported separately). Positive class for sensitivity = DC.")

    boot = st / "bootstrap_ci.csv"
    if boot.exists():
        b = pd.read_csv(boot)
        b = b[b.representation.isin(PRIMARY)]
        out = pd.DataFrame(dict(representation=b.description,
                                BA=[f"{v:.3f} [{l:.3f}, {h:.3f}]" for v, l, h in zip(b.ba, b.ba_ci_low, b.ba_ci_high)],
                                AUC=[f"{v:.3f} [{l:.3f}, {h:.3f}]" for v, l, h in zip(b.auc, b.auc_ci_low, b.auc_ci_high)]))
        add("table_bootstrap", "95% bootstrap confidence intervals (repeat 0)", out,
            "step4 bootstrap_ci.csv", b.method.iloc[0] if len(b) else "")

    # ---- ablation grid
    rows = []
    for m in ["HIST", "CBCT_U", "CBCT_L"]:
        for part in ["slic16", "grid16", "rand16"]:
            r = dict(modality=m, partition=part)
            for pool in ["mean", "max", "attn"]:
                rep = f"{m}|{part}_masked|{pool}"
                if rep in have:
                    r[f"BA {pool}"] = fmt(summ.loc[rep].balanced_accuracy_mean, summ.loc[rep].balanced_accuracy_sd)
                    r[f"AUC {pool}"] = fmt(summ.loc[rep].roc_auc_mean, summ.loc[rep].roc_auc_sd)
            rows.append(r)
        if f"{m}|whole|lr" in have:
            s = summ.loc[f"{m}|whole|lr"]
            rows.append(dict(modality=m, partition="none (whole image)",
                             **{"BA mean": fmt(s.balanced_accuracy_mean, s.balanced_accuracy_sd),
                                "AUC mean": fmt(s.roc_auc_mean, s.roc_auc_sd)}))
        if f"{m}|slic16_masked|concat16" in have:
            s = summ.loc[f"{m}|slic16_masked|concat16"]
            rows.append(dict(modality=m, partition="slic16 concatenated (original baseline)",
                             **{"BA mean": fmt(s.balanced_accuracy_mean, s.balanced_accuracy_sd),
                                "AUC mean": fmt(s.roc_auc_mean, s.roc_auc_sd)}))
    add("table_ablation", "Ablation: partition × pooling", pd.DataFrame(rows), "step3 summary.csv",
        "Identical frozen ResNet-18 features, folds and classifiers across rows.")

    rows = []
    for m in ["HIST", "CBCT_U", "CBCT_L"]:
        for k in ["slic8_masked", "slic16_masked", "slic32_masked", "slic16_m5_masked",
                  "slic16_m20_masked", "slic16_crop", "grid16_crop"]:
            r = dict(modality=m, setting=k)
            for pool in ["mean", "attn"]:
                rep = f"{m}|{k}|{pool}"
                if rep in have:
                    r[f"BA {pool}"] = fmt(summ.loc[rep].balanced_accuracy_mean, summ.loc[rep].balanced_accuracy_sd)
                    r[f"AUC {pool}"] = fmt(summ.loc[rep].roc_auc_mean, summ.loc[rep].roc_auc_sd)
            rows.append(r)
    add("table_sensitivity", "Sensitivity to region count, compactness and region input",
        pd.DataFrame(rows), "step3 summary.csv")

    # ---- statistics
    p = st / "permutation_tests.csv"
    if p.exists():
        d = pd.read_csv(p)
        add("table_permutation", "Full-pipeline permutation tests",
            d[["description", "n_permutations", "observed_ba", "null_ba_mean", "p_ba",
               "observed_auc", "null_auc_mean", "p_auc"]],
            "step4 permutation_tests.csv",
            f"Patient-level label permutation; entire pipeline refitted per permutation; one-sided; "
            f"p floor = 1/(N+1).")
    p = st / "paired_comparisons.csv"
    if p.exists() and p.stat().st_size > 5:
        d = pd.read_csv(p)
        cols = ["question", "model_a", "model_b", "delta_ba_mean_folds", "p_ba_corrected_t",
                "p_holm_ba_corrected_t", "delta_auc_mean_folds", "p_holm_auc_corrected_t",
                "delta_ba_boot_ci_low", "delta_ba_boot_ci_high", "mcnemar_p_median"]
        add("table_comparisons", "Pre-specified paired comparisons", d[[c for c in cols if c in d]],
            "step4 paired_comparisons.csv",
            "Primary test: Nadeau–Bengio corrected resampled t-test over all repeats × folds "
            "(two-sided), Holm-adjusted across the family. Bootstrap CI: paired stratified "
            "bootstrap of repeat-0 OOF predictions. McNemar: exact two-sided, median over repeats.")

    # ---- confusion matrices
    rows = []
    for rep in PRIMARY:
        g = oof[(oof.representation == rep) & (oof.repeat == 0)]
        if len(g):
            cm = confusion_matrix(g.label, g.pred, labels=list(range(n_classes)))
            for i, c in enumerate(classes):
                rows.append(dict(representation=describe(rep), true_class=c,
                                 **{f"pred_{cc}": int(cm[i, j]) for j, cc in enumerate(classes)}))
    add("table_confusion", "Confusion matrices (repeat 0, pooled out-of-fold)", pd.DataFrame(rows),
        "step3 oof_predictions.csv.gz")

    # ================================================================= figures
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    if n_classes == 2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, m in zip(axes, ["HIST", "CBCT_L", "MM_L"]):
            for k, lab in [("whole|lr", "whole image"), ("slic16_masked|mean", "SLIC + mean"),
                           ("slic16_masked|attn", "SLIC + attention")]:
                rep = f"{m}|{k}"
                g = oof[(oof.representation == rep) & (oof.repeat == 0)]
                if len(g):
                    fpr, tpr, _ = roc_curve(g.label, g.prob_1)
                    ax.plot(fpr, tpr, lw=1.6, label=f"{lab} (AUC {summ.loc[rep].roc_auc_mean:.3f})")
            ax.plot([0, 1], [0, 1], ls=":", c="grey"); ax.set_title(m)
            ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(fdir / "fig_roc.png"); plt.close(fig)
        inv.append(dict(item="fig_roc", title="ROC curves", source="repeat-0 pooled OOF",
                        note="legend AUC = mean over repeats"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, m in zip(axes, ["HIST", "CBCT_U", "CBCT_L"]):
        labels, means, sds = [], [], []
        for k in ["whole|lr", "slic16_masked|concat16"] + [f"{p}_masked|{q}" for p in
                                                           ["slic16", "grid16", "rand16"] for q in ["mean", "max", "attn"]]:
            rep = f"{m}|{k}"
            if rep in have:
                labels.append(k.replace("_masked", "").replace("|", " ")); means.append(summ.loc[rep].balanced_accuracy_mean)
                sds.append(summ.loc[rep].balanced_accuracy_sd)
        ax.errorbar(means, range(len(means)), xerr=sds, fmt="o", ms=4, capsize=2)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.axvline(0.5, ls=":", c="grey")
        ax.set_title(m); ax.set_xlabel("balanced accuracy (mean ± SD over repeats)")
    fig.tight_layout(); fig.savefig(fdir / "fig_ablation.png"); plt.close(fig)
    inv.append(dict(item="fig_ablation", title="Ablation", source="step3 summary.csv", note=""))

    nullf = st / "permutation_null_distributions.csv.gz"
    if nullf.exists():
        nd = pd.read_csv(nullf); pt = pd.read_csv(st / "permutation_tests.csv")
        reps = pt.representation.tolist()
        fig, axes = plt.subplots(1, len(reps), figsize=(3 * len(reps), 2.8), squeeze=False)
        for ax, rep in zip(axes[0], reps):
            ax.hist(nd[nd.representation == rep].ba, bins=30, color="#9bb")
            r = pt[pt.representation == rep].iloc[0]
            ax.axvline(r.observed_ba, c="crimson"); ax.set_title(rep.replace("|", "\n"), fontsize=7)
            ax.set_xlabel(f"BA (p={r.p_ba:.4f})")
        fig.tight_layout(); fig.savefig(fdir / "fig_permutation.png"); plt.close(fig)

    # partition examples and held-out attention maps
    att_f = cv / "attention_weights.csv.gz"
    att = pd.read_csv(att_f) if att_f.exists() else pd.DataFrame()
    examples = [cohort[cohort.label == c].patient_id.iloc[0] for c in range(min(n_classes, 2))]
    mods = ["HIST", "CBCT_U", "CBCT_L"]
    fig, axes = plt.subplots(len(examples) * len(mods), 4, figsize=(10, 2.6 * len(examples) * len(mods)))
    for i, pid in enumerate(examples):
        for j, mod in enumerate(mods):
            row = axes[i * len(mods) + j]
            with np.load(feature_path(mod, pid)) as z:
                img = z["preview"]; segs = {s: z[f"seg_{s}"] for s in ("slic16", "grid16", "rand16")}
            show = img if img.ndim == 3 else np.repeat(img[..., None], 3, 2)
            for ax, s in zip(row[:3], segs):
                b = find_boundaries(segs[s]); im2 = show.copy(); im2[b] = [255, 255, 0]
                ax.imshow(im2); ax.set_title(f"{pid} {mod} {s}", fontsize=7); ax.axis("off")
            ax = row[3]; ax.imshow(show); ax.axis("off")
            a = att[(att.representation == f"{mod}|slic16_masked|attn") & (att.patient_id == pid)]
            if len(a):
                heat = np.zeros(segs["slic16"].shape, float)
                for _, r in a.iterrows():
                    heat[segs["slic16"] == r.region] = r.weight
                ax.imshow(heat, alpha=0.5, cmap="magma"); ax.set_title("held-out attention", fontsize=7)
    fig.tight_layout(); fig.savefig(fdir / "fig_partitions_attention.png"); plt.close(fig)
    inv.append(dict(item="fig_partitions_attention", title="Partitions and held-out attention",
                    source="step2 features + step3 attention (repeat 0, fold model that did not see the patient)",
                    note=""))

    # ================================================================= write
    pd.DataFrame(inv).to_csv(rep_dir / "results_inventory.csv", index=False)
    save_json(dict(config=config_snapshot(), environment=environment_info()), rep_dir / "run_config.json")
    (rep_dir / "RESULTS_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    log(f"Report written to {rep_dir}")


if __name__ == "__main__":
    main()

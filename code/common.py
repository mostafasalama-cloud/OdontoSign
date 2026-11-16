"""Shared helpers: paths, devices, seeding, metrics, feature-store access."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

ID_RE = re.compile(r"^([A-Za-z]+)_(\d+)_", re.IGNORECASE)


# -----------------------------------------------------------------------------
# Output layout
# -----------------------------------------------------------------------------
def out_dir(*parts) -> Path:
    p = C.OUTPUT_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_dir(*parts) -> Path:
    """Results specific to the current TASK (and cohort variant)."""
    name = C.TASK + ("_original100" if C.RESTRICT_TO_ORIGINAL_100 else "")
    return out_dir("results", name, *parts)


def log(msg: str):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def parse_id(filename: str):
    """'AME_12_CBCT_Image.nii.gz' -> ('AME_12', 'AME', 12)."""
    m = ID_RE.match(Path(filename).name)
    if not m:
        return None
    cls, num = m.group(1).upper(), int(m.group(2))
    return f"{cls}_{num}", cls, num


# -----------------------------------------------------------------------------
# Reproducibility / devices
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def pick_device(pref: str):
    import torch
    if pref != "auto":
        return torch.device(pref)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": sys.platform}
    for mod in ["numpy", "scipy", "pandas", "sklearn", "skimage", "torch",
                "torchvision", "nibabel", "statsmodels", "PIL"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = "not installed"
    return info


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def config_snapshot() -> dict:
    keys = [k for k in dir(C) if k.isupper()]
    return {k: getattr(C, k) for k in keys}


# -----------------------------------------------------------------------------
# Cohort
# -----------------------------------------------------------------------------
def load_cohort() -> pd.DataFrame:
    path = out_dir("cohort") / "cohort_all.csv"
    if not path.exists():
        sys.exit("Run step1_build_cohort.py first.")
    df = pd.read_csv(path)
    classes = C.TASK_CLASSES[C.TASK]
    df = df[df["eligible"] & df["clinical_class"].isin(classes)].copy()
    if C.RESTRICT_TO_ORIGINAL_100:
        keep = (df["clinical_class"] == "AME") | (
            (df["clinical_class"] == "DC") & df["case_number"].isin(C.ORIGINAL_100_DC_IDS))
        df = df[keep]
    df["label"] = df["clinical_class"].map({c: i for i, c in enumerate(classes)})
    df = df.sort_values(["label", "case_number"]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Feature store
# -----------------------------------------------------------------------------
def feature_path(modality: str, pid: str) -> Path:
    return out_dir("features", modality) / f"{pid}.npz"


def load_bags(modality: str, key: str, pids):
    """Return list of (K_i, 512) arrays for feature key, e.g. 'slic16_masked'."""
    bags = []
    for pid in pids:
        with np.load(feature_path(modality, pid)) as z:
            bags.append(z[key].astype(np.float32))
    return bags


def load_pad_feature(modality: str) -> np.ndarray:
    with np.load(out_dir("features", modality) / "_zero_image_feature.npz") as z:
        return z["feature"].astype(np.float32)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_metrics(y_true, prob, n_classes) -> dict:
    """Metrics from class-probability matrix prob (n, C)."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    pred = prob.argmax(1)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
    }
    present = np.unique(y_true)
    try:
        if n_classes == 2:
            out["roc_auc"] = roc_auc_score(y_true, prob[:, 1]) if len(present) == 2 else np.nan
        else:
            out["roc_auc"] = (roc_auc_score(y_true, prob, multi_class="ovr", average="macro",
                                            labels=list(range(n_classes)))
                              if len(present) == n_classes else np.nan)
    except ValueError:
        out["roc_auc"] = np.nan
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        # positive class = label 1 (DC in the primary task)
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
        out["precision"] = tp / (tp + fp) if (tp + fp) else np.nan
        out["npv"] = tn / (tn + fn) if (tn + fn) else np.nan
        out["f1_positive"] = f1_score(y_true, pred, pos_label=1, zero_division=0)
    else:
        rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
        for i, r in enumerate(rec):
            out[f"recall_class{i}"] = r
    return out


def fast_ba(y, pred, n_classes):
    rec = []
    for c in range(n_classes):
        m = y == c
        if m.any():
            rec.append((pred[m] == c).mean())
    return float(np.mean(rec))


def fast_auc(y, prob, n_classes):
    """Rank-based AUC (binary) or macro one-vs-rest AUC; nan if undefined."""
    from scipy.stats import rankdata
    def bin_auc(t, s):
        n1 = t.sum(); n0 = len(t) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        r = rankdata(s)
        return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    if n_classes == 2:
        return bin_auc((y == 1).astype(int), prob[:, 1])
    vals = [bin_auc((y == c).astype(int), prob[:, c]) for c in range(n_classes)]
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else np.nan

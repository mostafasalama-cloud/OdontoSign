"""
Shared definitions for the CBCT-centred study (steps 6-9).

Representations are pre-specified here, BEFORE any result was seen.
Primary CBCT model (pre-specified): "RAD+C25_dinob"  = 3D lesion descriptors +
mean-pooled DINOv2-B 2.5D multi-plane lesion features, logistic regression.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import numpy as np

import config as CFG
from common import compute_metrics, out_dir

PRIMARY = "RAD+C25_dinob"

# name -> dict(parts=[(source, key, pool)], learner, alpha, group, label)
#   pool: vec (1-row bag), mean, max, bag (attention-MIL)
#   learner: lr | mil | mlp | kd | cnn3d (step 8) ; LATE(...) built post hoc
REGISTRY = {
    # ------------------------------------------------------------- CBCT only
    "C2D_lesion":   dict(parts=[("CBCT_L", "whole", "vec")], learner="lr", group="CBCT",
                         label="2D lesion summary, ResNet-18 (step 2)"),
    "RAD":          dict(parts=[("RAD", "rad", "vec")], learner="lr", group="CBCT",
                         label="3D lesion descriptors"),
    "C25_r18":      dict(parts=[("C25_r18", "planes", "mean")], learner="lr", group="CBCT",
                         label="2.5D multi-plane, ResNet-18, mean"),
    "C25_dinos":    dict(parts=[("C25_dinos", "planes", "mean")], learner="lr", group="CBCT",
                         label="2.5D multi-plane, DINOv2-S, mean"),
    "C25_dinob":    dict(parts=[("C25_dinob", "planes", "mean")], learner="lr", group="CBCT",
                         label="2.5D multi-plane, DINOv2-B, mean"),
    "C25_r18_attn": dict(parts=[("C25_r18", "planes", "bag")], learner="mil", group="CBCT",
                         label="2.5D multi-plane, ResNet-18, attention-MIL"),
    "C25_dinob_attn": dict(parts=[("C25_dinob", "planes", "bag")], learner="mil", group="CBCT",
                           label="2.5D multi-plane, DINOv2-B, attention-MIL"),
    "RAD+C25_dinos": dict(parts=[("RAD", "rad", "vec"), ("C25_dinos", "planes", "mean")],
                          learner="lr", group="CBCT", label="descriptors + DINOv2-S 2.5D"),
    "RAD+C25_dinob": dict(parts=[("RAD", "rad", "vec"), ("C25_dinob", "planes", "mean")],
                          learner="lr", group="CBCT", label="descriptors + DINOv2-B 2.5D (primary)"),
    "RAD+C25_dinob_MLP": dict(parts=[("RAD", "rad", "vec"), ("C25_dinob", "planes", "mean")],
                              learner="mlp", alpha=0.0, group="CBCT",
                              label="descriptors + DINOv2-B 2.5D, MLP"),
    "RAD+C25_dinob_KD": dict(parts=[("RAD", "rad", "vec"), ("C25_dinob", "planes", "mean")],
                             learner="kd", alpha=0.5, teacher=[("HT_dinob", "tiles", "mean")],
                             group="CBCT", label="descriptors + DINOv2-B 2.5D, MLP + histology distillation"),
    "CNN3D":        dict(parts=[], learner="cnn3d", group="CBCT", label="3D CNN on lesion cube (trained per fold)"),
    # ---------------------------------------------------- histopathology (upper bound)
    "H_whole_r18":  dict(parts=[("HIST", "whole", "vec")], learner="lr", group="HIST",
                         label="whole image 224 px, ResNet-18 (step 2)"),
    "HT_r18":       dict(parts=[("HT_r18", "tiles", "mean")], learner="lr", group="HIST",
                         label="ROI tiles, ResNet-18, mean"),
    "HT_dinos":     dict(parts=[("HT_dinos", "tiles", "mean")], learner="lr", group="HIST",
                         label="ROI tiles, DINOv2-S, mean"),
    "HT_dinob":     dict(parts=[("HT_dinob", "tiles", "mean")], learner="lr", group="HIST",
                         label="ROI tiles, DINOv2-B, mean"),
    "HT_dinob_attn": dict(parts=[("HT_dinob", "tiles", "bag")], learner="mil", group="HIST",
                          label="ROI tiles, DINOv2-B, attention-MIL"),
    # ------------------------------------------------------------ multimodal
    "MM_RAD+C25+HT": dict(parts=[("RAD", "rad", "vec"), ("C25_dinob", "planes", "mean"),
                                 ("HT_dinob", "tiles", "mean")], learner="lr", group="MM",
                          label="CBCT (descriptors + DINOv2-B 2.5D) + histology tiles DINOv2-B"),
}
LATE = {
    "LATE(RAD+C25_dinob,CNN3D)": ["RAD+C25_dinob", "CNN3D"],
    "LATE(RAD,C25_dinob)": ["RAD", "C25_dinob"],
    "LATE(CBCT,HIST)": ["RAD+C25_dinob", "HT_dinob"],
}
LATE_LABELS = {
    "LATE(RAD+C25_dinob,CNN3D)": "late fusion: primary + 3D CNN",
    "LATE(RAD,C25_dinob)": "late fusion: descriptors + DINOv2-B 2.5D",
    "LATE(CBCT,HIST)": "late fusion: primary CBCT + histology tiles",
}
GROUP = {**{k: v["group"] for k, v in REGISTRY.items()}, "LATE(RAD+C25_dinob,CNN3D)": "CBCT",
         "LATE(RAD,C25_dinob)": "CBCT", "LATE(CBCT,HIST)": "MM"}


def label(name):
    return REGISTRY[name]["label"] if name in REGISTRY else LATE_LABELS.get(name, name)


def study_dir(*parts):
    name = CFG.TASK + ("_original100" if CFG.RESTRICT_TO_ORIGINAL_100 else "")
    return out_dir("results", name, "cbct_study", *parts)


# =============================================================================
# Data assembly
# =============================================================================
def _load(src, key, pids):
    bags = []
    for pid in pids:
        with np.load(out_dir("features", src) / f"{pid}.npz") as z:
            bags.append(z[key].astype(np.float32))
    return bags


def _vectors(parts, pids):
    cols = []
    for src, key, pool in parts:
        bags = _load(src, key, pids)
        if pool in ("vec", "mean"):
            cols.append(np.stack([b.mean(0) for b in bags]))
        elif pool == "max":
            cols.append(np.stack([b.max(0) for b in bags]))
    return np.hstack(cols).astype(np.float32)


def assemble(name, pids):
    spec = REGISTRY[name]
    if spec["learner"] == "mil":
        Xs, Ms = [], []
        for src, key, _ in spec["parts"]:
            bags = _load(src, key, pids)
            k = max(len(b) for b in bags); d = bags[0].shape[1]
            X = np.zeros((len(bags), k, d), np.float32); M = np.zeros((len(bags), k), bool)
            for i, b in enumerate(bags):
                X[i, :len(b)] = b; M[i, :len(b)] = True
            Xs.append(X); Ms.append(M)
        return dict(X=Xs, M=Ms)
    data = dict(X=_vectors(spec["parts"], pids))
    if spec["learner"] == "kd":
        data["T"] = _vectors(spec["teacher"], pids)
    return data


def subset(data, idx):
    if "M" in data:
        return dict(X=[x[idx] for x in data["X"]], M=[m[idx] for m in data["M"]])
    return {k: v[idx] for k, v in data.items()}


# =============================================================================
# Learners (everything fitted on the training fold only)
# =============================================================================
def _lr_pipeline(seed, ytr):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    n_inner = min(CFG.LR_INNER_FOLDS, int(np.bincount(ytr).min()))
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        FunctionTransformer(lambda a: np.clip(a, -5, 5)),
        LogisticRegressionCV(Cs=CFG.LR_CS, cv=StratifiedKFold(n_inner, shuffle=True, random_state=seed),
                             class_weight="balanced", scoring="neg_log_loss", max_iter=5000, n_jobs=1))


def fit_lr(tr, ytr, te, n_classes, seed):
    import warnings
    clf = _lr_pipeline(seed, ytr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(tr["X"], ytr)
    return clf.predict_proba(te["X"]), dict(C=float(np.ravel(clf[-1].C_)[0]))


def teacher_soft_labels(T, y, n_classes, seed):
    """Out-of-sample teacher probabilities for the TRAINING patients (inner cross-fitting)."""
    import warnings
    from sklearn.model_selection import StratifiedKFold
    P = np.zeros((len(y), n_classes), np.float32)
    k = min(5, int(np.bincount(y).min()))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(T, y):
        clf = _lr_pipeline(seed, y[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(T[tr], y[tr])
        P[te] = clf.predict_proba(T[te])
    return P


def fit_mlp(tr, ytr, te, n_classes, seed, alpha=0.0):
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    imp = SimpleImputer(strategy="median").fit(tr["X"])
    sc = StandardScaler().fit(imp.transform(tr["X"]))
    f = lambda a: torch.tensor(np.clip(sc.transform(imp.transform(a)), -5, 5), dtype=torch.float32)
    Xtr, Xte = f(tr["X"]), f(te["X"])
    y = torch.tensor(ytr, dtype=torch.long)
    cnt = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(len(ytr) / (n_classes * np.maximum(cnt, 1)))
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
                              torch.nn.Dropout(0.3), torch.nn.Linear(128, n_classes))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    ce = torch.nn.CrossEntropyLoss(weight=w)
    if alpha > 0:
        Pt = torch.tensor(teacher_soft_labels(tr["T"], ytr, n_classes, seed)).clamp_min(1e-6)
    net.train()
    for _ in range(300):  # fixed, full batch
        opt.zero_grad()
        logits = net(Xtr)
        loss = ce(logits, y)
        if alpha > 0:
            kl = torch.nn.functional.kl_div(torch.log_softmax(logits, 1), Pt, reduction="batchmean")
            loss = (1 - alpha) * loss + alpha * kl
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(Xte), 1).numpy(), {}


def fit_predict(name, tr, ytr, te, n_classes, seed):
    spec = REGISTRY[name]
    if spec["learner"] == "lr":
        return fit_lr(tr, ytr, te, n_classes, seed)
    if spec["learner"] in ("mlp", "kd"):
        return fit_mlp(tr, ytr, te, n_classes, seed, spec.get("alpha", 0.0))
    if spec["learner"] == "mil":
        from models import fit_predict_mil
        prob, _, extra = fit_predict_mil(tr, ytr, te, n_classes, seed)
        return prob, extra
    raise ValueError(spec["learner"])


def run_cv(name, data, y, splits, n_classes, repeat=0, fit=None):
    fit = fit or fit_predict
    prob_oof = np.full((len(y), n_classes), np.nan, np.float32)
    rows = []
    for f, (tr, te) in enumerate(splits):
        prob, extra = fit(name, subset(data, tr), y[tr], subset(data, te), n_classes,
                          CFG.SEED + 1000 * repeat + f)
        prob_oof[te] = prob
        rows.append(dict(repeat=repeat, fold=f, n_train=len(tr), n_test=len(te),
                         **compute_metrics(y[te], prob, n_classes), **extra))
    return prob_oof, rows

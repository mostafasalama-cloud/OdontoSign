"""
Representation registry, data assembly and the two learners used inside every
training fold:

  * Logistic regression (for label-free pooled / whole-image / concatenated
    features). Standardisation and the choice of C (inner stratified 5-fold CV)
    are fitted on the training fold only.
  * Attention-MIL (the proposed aggregation). Region-feature standardisation,
    the attention network and the classifier head are trained jointly,
    end-to-end, on the training fold only; held-out patients are only ever
    passed forward through the frozen fold model.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embedded-style Python (._pth) does not add the script folder
import numpy as np

import config as CFG
from common import compute_metrics, load_bags, load_pad_feature

BASE_MODS = CFG.CBCT_MODES + ["HIST"]
MM = {"MM_U": ["CBCT_U", "HIST"], "MM_L": ["CBCT_L", "HIST"]}


# =============================================================================
# Registry
# =============================================================================
def representation_registry():
    reps = []
    main_keys = ["slic16_masked", "grid16_masked", "rand16_masked", "slic16_crop", "grid16_crop"]
    sens_keys = ["slic8_masked", "slic32_masked", "slic16_m5_masked", "slic16_m20_masked"]
    for m in BASE_MODS:
        reps.append(f"{m}|whole|lr")
        reps.append(f"{m}|slic16_masked|concat16")
        for k in main_keys:
            reps += [f"{m}|{k}|mean", f"{m}|{k}|max", f"{m}|{k}|attn"]
        for k in sens_keys:
            reps += [f"{m}|{k}|mean", f"{m}|{k}|attn"]
    for mm in MM:
        reps += [f"{mm}|whole|lr", f"{mm}|slic16_masked|concat16", f"{mm}|slic16_masked|mean",
                 f"{mm}|slic16_masked|attn", f"{mm}|grid16_masked|attn", f"{mm}|slic16_crop|attn"]
    return reps


def parse_rep(rep):
    mod, key, pool = rep.split("|")
    mods = MM.get(mod, [mod])
    learner = "mil" if pool == "attn" else "lr"
    return mods, key, pool, learner


def describe(rep):
    mod, key, pool = rep.split("|")
    names = {"CBCT_U": "CBCT (uniform slices)", "CBCT_L": "CBCT (lesion ROI)",
             "HIST": "Histopathology", "MM_U": "Multimodal (CBCT uniform + Hist)",
             "MM_L": "Multimodal (CBCT lesion + Hist)"}
    pools = {"lr": "whole-image feature + LR", "mean": "mean pooling + LR",
             "max": "max pooling + LR", "concat16": "concatenation of 16 regions + LR",
             "attn": "attention-MIL"}
    part = "" if key == "whole" else f" · {key.replace('_', ' ')}"
    return f"{names[mod]}{part} · {pools[pool]}"


# =============================================================================
# Data assembly
# =============================================================================
_cache = {}


def _bags(mod, key, pids):
    ck = (mod, key, tuple(pids))
    if ck not in _cache:
        _cache[ck] = load_bags(mod, key, pids)
    return _cache[ck]


def assemble(rep, pids):
    """LR -> dict(X=(n,d)); MIL -> dict(X=[(n,Kmax,512)...], M=[(n,Kmax) bool...])."""
    mods, key, pool, learner = parse_rep(rep)
    if learner == "lr":
        parts = []
        for m in mods:
            bags = _bags(m, key, pids)
            if pool in ("lr", "mean"):
                v = np.stack([b.mean(0) for b in bags])
            elif pool == "max":
                v = np.stack([b.max(0) for b in bags])
            elif pool == "concat16":
                pad = load_pad_feature(m)
                rows = []
                for b in bags:  # first K regions in SLIC label order, zero-image padding
                    b = b[:CFG.CONCAT_K]
                    if len(b) < CFG.CONCAT_K:
                        b = np.vstack([b, np.repeat(pad[None], CFG.CONCAT_K - len(b), 0)])
                    rows.append(b.reshape(-1))
                v = np.stack(rows)
            else:
                raise ValueError(pool)
            parts.append(v)
        return dict(X=np.hstack(parts).astype(np.float32))
    Xs, Ms = [], []
    for m in mods:
        bags = _bags(m, key, pids)
        kmax = max(len(b) for b in bags)
        X = np.zeros((len(bags), kmax, CFG.FEATURE_DIM), np.float32)
        M = np.zeros((len(bags), kmax), bool)
        for i, b in enumerate(bags):
            X[i, :len(b)] = b
            M[i, :len(b)] = True
        Xs.append(X); Ms.append(M)
    return dict(X=Xs, M=Ms)


def subset(data, idx):
    if "M" in data:
        return dict(X=[x[idx] for x in data["X"]], M=[m[idx] for m in data["M"]])
    return dict(X=data["X"][idx])


# =============================================================================
# Learners
# =============================================================================
def fit_predict_lr(tr, ytr, te, n_classes, seed):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_inner = min(CFG.LR_INNER_FOLDS, int(np.bincount(ytr).min()))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=CFG.LR_CS,
                             cv=StratifiedKFold(n_inner, shuffle=True, random_state=seed),
                             class_weight="balanced", scoring="neg_log_loss",
                             max_iter=5000, n_jobs=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", UserWarning)
        clf.fit(tr["X"], ytr)
    prob = clf.predict_proba(te["X"])
    return prob, None, dict(C=float(np.ravel(clf[-1].C_)[0]))


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def build_mil(n_branches, n_classes, dims=None):
    dims = dims or [CFG.FEATURE_DIM] * n_branches
    torch = _torch()
    nn = torch.nn
    p = CFG.MIL

    class AttnMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = nn.ModuleList([
                nn.Sequential(nn.Linear(d, p["att_hidden"]), nn.Tanh(),
                              nn.Linear(p["att_hidden"], 1)) for d in dims])
            h1, h2 = p["head"]
            self.head = nn.Sequential(
                nn.Linear(sum(dims), h1), nn.ReLU(), nn.Dropout(p["dropout"][0]),
                nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p["dropout"][1]),
                nn.Linear(h2, n_classes))

        def forward(self, Xs, Ms):
            zs, atts = [], []
            for att, X, M in zip(self.att, Xs, Ms):
                s = att(X).squeeze(-1).masked_fill(~M, float("-inf"))
                a = torch.softmax(s, dim=1)                   # eq. (9), padded regions = 0
                zs.append(torch.bmm(a.unsqueeze(1), X).squeeze(1))  # eq. (10)
                atts.append(a)
            return self.head(torch.cat(zs, 1)), atts

    return AttnMIL()


def fit_predict_mil(tr, ytr, te, n_classes, seed):
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(CFG.MIL_DEVICE)

    # region-feature standardisation fitted on valid TRAINING regions only
    Xtr, Xte = [], []
    for X, M, X2, M2 in zip(tr["X"], tr["M"], te["X"], te["M"]):
        valid = X[M]
        mu, sd = valid.mean(0), valid.std(0) + 1e-6
        Xtr.append(torch.tensor((X - mu) / sd * keep(M), device=dev))
        Xte.append(torch.tensor((X2 - mu) / sd * keep(M2), device=dev))
    Mtr = [torch.tensor(m, device=dev) for m in tr["M"]]
    Mte = [torch.tensor(m, device=dev) for m in te["M"]]
    y = torch.tensor(ytr, dtype=torch.long, device=dev)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    w = torch.tensor(len(ytr) / (n_classes * np.maximum(counts, 1)), device=dev)  # balanced
    model = build_mil(len(Xtr), n_classes, [x.shape[-1] for x in Xtr]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.MIL["lr"],
                           weight_decay=CFG.MIL["weight_decay"])
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(CFG.MIL["epochs"]):  # full batch, fixed epochs, no early stopping
        opt.zero_grad()
        logits, _ = model(Xtr, Mtr)
        lossf(logits, y).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        logits, atts = model(Xte, Mte)
        prob = torch.softmax(logits, 1).cpu().numpy()
    return prob, [a.cpu().numpy() for a in atts], {}


def keep(M):
    return M[..., None].astype(np.float32)  # padded rows stay exactly zero


def fit_predict(learner, *args):
    return (fit_predict_mil if learner == "mil" else fit_predict_lr)(*args)


def fold_seed(repeat, fold):
    return CFG.SEED + 1000 * repeat + fold


# =============================================================================
# One full cross-validation pass of one representation (used by step3 and the
# permutation test in step4 -- identical procedure for observed and null runs)
# =============================================================================
def run_cv(rep, data, y, splits, n_classes, repeat=0, keep_attention=False):
    """splits: list of (train_idx, test_idx). Returns (prob_oof, fold_rows, attention)."""
    _, _, _, learner = parse_rep(rep)
    prob_oof = np.full((len(y), n_classes), np.nan, np.float32)
    fold_rows, attention = [], []
    for f, (tr_i, te_i) in enumerate(splits):
        prob, atts, extra = fit_predict(learner, subset(data, tr_i), y[tr_i],
                                        subset(data, te_i), n_classes, fold_seed(repeat, f))
        prob_oof[te_i] = prob
        m = compute_metrics(y[te_i], prob, n_classes)
        fold_rows.append(dict(repeat=repeat, fold=f, n_train=len(tr_i), n_test=len(te_i),
                              **{f"n_test_class{c}": int((y[te_i] == c).sum())
                                 for c in range(n_classes)}, **m, **extra))
        if keep_attention and atts is not None:
            attention.append((te_i, atts))
    return prob_oof, fold_rows, attention

"""
STEP 8 (CBCT-centred study): a compact 3D CNN trained from scratch inside every
training fold on the lesion-centred cube (2 channels: CBCT intensity + expert
lesion mask, 64^3). Fixed recipe chosen a priori (no tuning on test folds,
no early stopping): AdamW 1e-3, weight decay 1e-2, cosine schedule, 80 epochs,
batch 16, class-balanced cross-entropy, flips / axial 90-degree rotations /
intensity jitter / noise. Uses the first N_CNN_REPEATS repeats of the same
splits as all other models. Results go to the step-7 cache as "CNN3D".
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import pickle
import time

import numpy as np

import config as CFG
from cbct_study import study_dir
from common import compute_metrics, load_cohort, log, out_dir, pick_device
from step3_evaluate import make_splits
from step7_cbct_evaluate import run_hash

N_CNN_REPEATS = int(_os.environ.get("DOLCHID_CNN_REPEATS", "3"))
EPOCHS = int(_os.environ.get("DOLCHID_CNN_EPOCHS", "80"))
BATCH = 16


def build(n_classes):
    import torch.nn as nn

    def block(i, o, s):
        return nn.Sequential(nn.Conv3d(i, o, 3, s, 1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
                             nn.Conv3d(o, o, 3, 1, 1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
    return nn.Sequential(block(2, 16, 2), block(16, 32, 2), block(32, 64, 2), block(64, 128, 2),
                         nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))


def augment(x, rng):
    x = x.copy()
    for ax in (1, 2, 3):
        if rng.random() < 0.5:
            x = np.flip(x, ax)
    x = np.rot90(x, rng.integers(4), axes=(1, 2))
    x[0] = x[0] * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05) + rng.normal(0, 0.02, x[0].shape)
    return np.ascontiguousarray(x)


def train_predict(X, ytr, Xte, n_classes, seed, dev):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = build(n_classes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = EPOCHS * int(np.ceil(len(ytr) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    cnt = np.bincount(ytr, minlength=n_classes)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (n_classes * np.maximum(cnt, 1)),
                                                          dtype=torch.float32, device=dev))
    net.train()
    for _ in range(EPOCHS):
        order = rng.permutation(len(ytr))
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            xb = torch.from_numpy(np.stack([augment(X[j], rng) for j in b])).float().to(dev)
            yb = torch.from_numpy(ytr[b]).long().to(dev)
            opt.zero_grad()
            lossf(net(xb), yb).backward()
            opt.step(); sched.step()
    net.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xte), BATCH):
            xb = torch.from_numpy(Xte[i:i + BATCH].astype(np.float32)).to(dev)
            # test-time augmentation: identity + 3 flips
            p = sum(torch.softmax(net(torch.flip(xb, d) if d else xb), 1) for d in (None, [2], [3], [4])) / 4
            probs.append(p.float().cpu().numpy())
    return np.concatenate(probs)


def main():
    import torch
    cohort = load_cohort()
    pids, y = cohort.patient_id.tolist(), cohort.label.to_numpy()
    n_classes = len(CFG.TASK_CLASSES[CFG.TASK])
    splits = make_splits(y)
    h = run_hash(pids)
    cf = study_dir("cache") / "CNN3D.pkl"
    prev = {}
    if cf.exists():
        with open(cf, "rb") as f:
            obj = pickle.load(f)
        if obj.get("hash") == h and obj.get("epochs") == EPOCHS:
            prev = {r: (r, p, rows) for r, p, rows in obj["results"]}
    todo = [r for r in range(min(N_CNN_REPEATS, CFG.N_REPEATS)) if r not in prev]
    if not todo:
        log("3D CNN: cached"); return
    dev = pick_device(CFG.FEATURE_DEVICE)
    log(f"3D CNN on {dev}: repeats {todo}, {EPOCHS} epochs")
    X = np.stack([np.load(out_dir("features", "CBCT3D") / f"{p}.npz")["cube"] for p in pids]).astype(np.float16)
    for r in todo:
        prob = np.full((len(y), n_classes), np.nan, np.float32); rows = []
        for f, (tr, te) in enumerate(splits[r]):
            t0 = time.time()
            try:
                p = train_predict(X[tr], y[tr], X[te], n_classes, CFG.SEED + 1000 * r + f, dev)
            except RuntimeError as e:  # e.g. an op not supported on the GPU backend
                if dev.type == "cpu":
                    raise
                log(f"  {dev} failed ({str(e)[:120]}); falling back to CPU")
                dev = torch.device("cpu")
                p = train_predict(X[tr], y[tr], X[te], n_classes, CFG.SEED + 1000 * r + f, dev)
            prob[te] = p
            m = compute_metrics(y[te], p, n_classes)
            rows.append(dict(repeat=r, fold=f, n_train=len(tr), n_test=len(te), **m))
            log(f"  repeat {r} fold {f}: BA={m['balanced_accuracy']:.3f} ({time.time() - t0:.0f}s)")
        prev[r] = (r, prob, rows)
        with open(cf, "wb") as fh:  # save after every repeat (resumable)
            pickle.dump(dict(hash=h, epochs=EPOCHS, results=[prev[k] for k in sorted(prev)]), fh)
    log("Step 8 complete.")


if __name__ == "__main__":
    main()

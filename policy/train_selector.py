"""M5 Step 3: the grasp selector -- one pre-registered MLP that predicts, for a (layout, candidate) pair, whether the b_hybrid_phi
run succeeds. Runs in hoi-perception (PyTorch); the data comes from hoi-sim as a plain npz, the model goes back as ONNX.

Input: results/m5/train_data.npz (policy/m5.py candidates): one row per (training layout, candidate) rollout: layout index,
clip index, candidate label, phi, phi_h, cup xy, coaster xy, handle yaw, success. Features: policy/features.py.

Model (pre-registered, no variants, no tuning): MLP 14 -> 64 -> 64 -> 1 logit, ReLU, BCE (BCEWithLogitsLoss), Adam lr 1e-3,
weight decay 1e-4, 300 epochs, full batch, torch seed 0, CPU, float32. Training inputs: 5 noisy copies of every training row
(policy/features.py noisy, one draw per copy over all rows, np.random.default_rng(0)); standardised with the mean / std of those
training features (std 0 -> 1).

Cross-validation: 5 folds grouped by layout (np.random.default_rng(0).permutation of the layouts, np.array_split into 5). Each
fold trains a fresh model as above on the other 4 folds and predicts its held-out layouts with NOISY features: one draw per
layout (np.random.default_rng(1), layouts in index order, all of a layout's candidates share it), the same draw whichever fold
holds the layout out. Reported on the out-of-fold predictions: AUC (rank / Mann-Whitney, ties averaged) and Brier score, and the
held-out selection success of R1, R2, F (recomputed on each training fold) and CEIL (any candidate succeeds), read from the
rollouts themselves (every candidate ran on every layout). Rules: policy/rules.py.

Final model: trained as above on all training layouts; exported with torch.onnx.export(..., dynamo=False) as one graph that takes
the raw features (n, 14) float32 and returns p (n, 1) = sigmoid(logit) (the standardisation is inside). 1000 saved feature rows
(the first 1000 clean training rows) with the torch p go to results/m5/onnx_check_rows.npz for the onnxruntime check in hoi-sim.

Run: env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m policy.train_selector
  -> results/m5/cv.json, results/m5/selector.onnx, results/m5/selector_meta.json, results/m5/onnx_check_rows.npz
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from policy.features import CLIPS, FEATURES, features, noisy
from policy.rules import fixed_best, r1, r2

OUT = Path("results/m5")
N_COPIES, EPOCHS, LR, WD, HIDDEN, N_FOLDS = 5, 300, 1e-3, 1e-4, 64, 5
TRAIN_NOISE_SEED, HELDOUT_NOISE_SEED, FOLD_SEED, TORCH_SEED = 0, 1, 0, 0
N_ONNX_ROWS = 1000


class Selector(torch.nn.Module):
    """Standardise, MLP, sigmoid: raw features (n, 14) -> p (n, 1)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(len(FEATURES), HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, HIDDEN),
                                       torch.nn.ReLU(), torch.nn.Linear(HIDDEN, 1))

    def logit(self, x):
        return self.mlp((x - self.mean) / self.std)

    def forward(self, x):
        return torch.sigmoid(self.logit(x))


def load(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def rows_features(d: dict, idx: np.ndarray, cup=None, coaster=None, yaw=None) -> np.ndarray:
    cup = d["cup_xy"][idx] if cup is None else cup
    coaster = d["coaster_xy"][idx] if coaster is None else coaster
    yaw = d["yaw"][idx] if yaw is None else yaw
    return features(cup, coaster, yaw, d["clip_index"][idx], d["phi"][idx], d["phi_h"][idx])


def train(d: dict, idx: np.ndarray) -> tuple[Selector, dict]:
    """The pre-registered model on rows ``idx`` (5 noisy copies, seed 0)."""
    rng = np.random.default_rng(TRAIN_NOISE_SEED)
    X = np.concatenate([rows_features(d, idx, *noisy(rng, d["cup_xy"][idx], d["coaster_xy"][idx], d["yaw"][idx])) for _ in range(N_COPIES)])
    y = np.tile(d["success"][idx].astype(np.float32), N_COPIES)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    torch.manual_seed(TORCH_SEED)
    model = Selector(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)[:, None]
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model.logit(Xt), yt)
        loss.backward()
        opt.step()
    return model, dict(rows=int(len(idx)), train_rows_with_noise=int(len(X)), final_loss=float(loss.item()))


def predict(model: Selector, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()[:, 0].astype(float)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def candidate_table(d: dict) -> list[tuple[int, str]]:
    """The candidates as (clip index, label) in table order (clip order, then the order they appear in the data)."""
    seen = []
    for c, k in zip(d["clip_index"].tolist(), d["candidate"].tolist()):
        if (c, k) not in seen:
            seen.append((c, k))
    return sorted(seen, key=lambda ck: (ck[0], seen.index(ck)))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="reads train_data.npz from here and writes the outputs here")
    out_dir = ap.parse_args().out
    t0 = time.perf_counter()
    d = load(out_dir / "train_data.npz")
    layouts = np.unique(d["layout_index"])
    cands = candidate_table(d)
    key = {ck: j for j, ck in enumerate(cands)}
    n_l, n_c = len(layouts), len(cands)
    # success / phi / departure / row index per (layout, candidate): every candidate ran on every layout (asserted)
    S = np.full((n_l, n_c), -1)
    row_of = np.full((n_l, n_c), -1)
    for i, (l, c, k) in enumerate(zip(d["layout_index"], d["clip_index"], d["candidate"])):
        S[np.searchsorted(layouts, l), key[(int(c), str(k))]] = int(d["success"][i])
        row_of[np.searchsorted(layouts, l), key[(int(c), str(k))]] = i
    assert (S >= 0).all(), "some (layout, candidate) rollout is missing"
    phi_c = np.array([d["phi"][row_of[0, j]] for j in range(n_c)])
    dep_c = np.array([d["phi_h"][row_of[0, j]] - d["phi"][row_of[0, j]] for j in range(n_c)])

    # held-out noise: one draw per layout, layouts in index order, shared by its candidates
    rng = np.random.default_rng(HELDOUT_NOISE_SEED)
    first = row_of[:, 0]
    ncup, ncoast, nyaw = noisy(rng, d["cup_xy"][first], d["coaster_xy"][first], d["yaw"][first])
    folds = np.array_split(np.random.default_rng(FOLD_SEED).permutation(n_l), N_FOLDS)
    oof_p = np.full((n_l, n_c), np.nan)
    fold_info = []
    for f, held in enumerate(folds):
        train_l = np.setdiff1d(np.arange(n_l), held)
        model, info = train(d, row_of[train_l].reshape(-1))
        rows = row_of[held].reshape(-1)
        li = np.repeat(held, n_c)
        X = rows_features(d, rows, ncup[li], ncoast[li], nyaw[li])
        oof_p[held] = predict(model, X).reshape(len(held), n_c)
        F = fixed_best(S[train_l].mean(axis=0), phi_c)
        sel = dict(R1=[S[l, r1(oof_p[l])] for l in held], R2=[S[l, r2(oof_p[l], dep_c)] for l in held], F=[S[l, F] for l in held],
                   CEIL=[int(S[l].max()) for l in held])
        fold_info.append(dict(fold=f, held_out_layouts=sorted(int(layouts[l]) for l in held), F_candidate=dict(clip=CLIPS[cands[F][0]], candidate=cands[F][1]),
                              **{k: int(sum(v)) for k, v in sel.items()}, n=int(len(held)), **info))
        print(f"fold {f}: " + ", ".join(f"{k} {sum(v)}/{len(held)}" for k, v in sel.items()) + f", F = {CLIPS[cands[F][0]]} {cands[F][1]}", flush=True)
    y, p = S.reshape(-1), oof_p.reshape(-1)
    cv = dict(auc=auc(y, p), brier=float(np.mean((p - y) ** 2)), base_rate=float(y.mean()), n_rows=int(len(y)), n_layouts=int(n_l),
              selection={k: int(sum(fi[k] for fi in fold_info)) for k in ("R1", "R2", "F", "CEIL")}, folds=fold_info)

    # final model on every training layout, ONNX export, check rows
    model, info = train(d, row_of.reshape(-1))
    onnx_path = out_dir / "selector.onnx"
    model.eval()
    torch.onnx.export(model, torch.zeros(2, len(FEATURES)), str(onnx_path), input_names=["features"], output_names=["p"],
                      dynamic_axes={"features": {0: "n"}, "p": {0: "n"}}, dynamo=False, opset_version=17)
    Xc = rows_features(d, np.arange(min(N_ONNX_ROWS, len(d["success"]))))
    np.savez(out_dir / "onnx_check_rows.npz", features=Xc.astype(np.float32), p_torch=predict(model, Xc.astype(np.float32)))
    meta = dict(what="M5 Step 3 selector (policy/train_selector.py)", features=list(FEATURES), candidates=[dict(clip=CLIPS[c], candidate=k) for c, k in cands],
                model=f"MLP {len(FEATURES)}-{HIDDEN}-{HIDDEN}-1 ReLU, BCEWithLogits, Adam lr {LR}, weight decay {WD}, {EPOCHS} epochs full batch, torch seed {TORCH_SEED}",
                noise=f"training: {N_COPIES} noisy copies per row, default_rng({TRAIN_NOISE_SEED}); held-out CV: one draw per layout, default_rng({HELDOUT_NOISE_SEED})",
                standardisation=dict(mean=model.mean.tolist(), std=model.std.tolist()), final=info, onnx=str(onnx_path), opset=17,
                torch=torch.__version__, numpy=np.__version__)
    (out_dir / "selector_meta.json").write_text(json.dumps(meta, indent=2))
    cv.update(wall_s=round(time.perf_counter() - t0, 1), torch=torch.__version__, threads=torch.get_num_threads())
    (out_dir / "cv.json").write_text(json.dumps(cv, indent=2))
    print(f"CV: AUC {cv['auc']:.3f}, Brier {cv['brier']:.3f} (base rate {cv['base_rate']:.3f}); selection " +
          ", ".join(f"{k} {v}/{n_l}" for k, v in cv["selection"].items()) + f"; wrote {onnx_path} ({cv['wall_s']} s)")


if __name__ == "__main__":
    main()

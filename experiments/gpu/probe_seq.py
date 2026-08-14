"""Structure-aware probe: train a small ATTENTION-POOLING decoder on the FULL
frame-level (time x dim) audio-encoder representation, instead of the time-mean
pooled vector every other probe (probe.py/probe_mlp.py) uses.

Motivation (2026-08-13): mean pooling collapses the time axis and can destroy
structure a global-but-temporally-encoded property (mode/key) might live in. The
NEVER_CAPTURED / LATE_LAYER_LOSS verdicts from probe.py/probe_mlp.py were all on
mean-pooled features -- this asks whether a decoder that KEEPS the sequence and
learns which frames to attend to recovers signal mean pooling missed.

Decoder: per-frame attention scores -> softmax over (non-silent) frames ->
weighted sum -> small MLP head. Held-out by soundfont (GroupKFold), same
discipline as probe.py. Reports mean CV accuracy per saved frame-level layer,
alongside the plain mean-pool baseline trained the same way, for a direct diff.

  CUDA_VISIBLE_DEVICES=6 python gpu/probe_seq.py --acts acts/mert330 \
      --task mode_id --target ground_truth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musicprobe.config import MANIFEST_PATH  # noqa: E402

MAX_T = 512          # cap frames (evenly subsample if longer) -- bounds memory/compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _facts(f):
    return f if isinstance(f, dict) else json.loads(f)


def _prep_frames(fr: np.ndarray) -> np.ndarray:
    """Drop near-silent/padding frames (Whisper/AF3 pad to 30s), cap length."""
    fr = fr.astype(np.float32)
    norm = np.linalg.norm(fr, axis=1)
    keep = norm > 1e-3 * (norm.max() + 1e-9)
    fr = fr[keep] if keep.sum() >= 4 else fr
    if len(fr) > MAX_T:
        idx = np.linspace(0, len(fr) - 1, MAX_T).astype(int)
        fr = fr[idx]
    return fr


def _frame_layers(z):
    return sorted(int(k.split("_")[1]) for k in z.files if k.startswith("layer_"))


def load_task(acts_dir: Path, man: pd.DataFrame, task: str, target: str,
              group_key: str, layer: int):
    sub = man[man.task == task]
    seqs, y, g = [], [], []
    key = f"layer_{layer:02d}"
    for r in sub.itertuples():
        p = acts_dir / (r.stimulus_id.replace("/", "__") + ".npz")
        if not p.exists():
            continue
        z = np.load(p)
        if key not in z.files:
            return None, None, None
        seqs.append(_prep_frames(z[key]))
        f = _facts(r.factors)
        y.append(f.get(target, r.ground_truth))
        g.append(f.get(group_key, "na"))
    return seqs, np.array(y), np.array(g)


class AttnPool(nn.Module):
    """Learnable attention pooling + LINEAR head. Attention scorer is init to
    ZERO -> uniform attention == mean pooling at start, so it can only improve
    on (never underperform) the mean baseline as it learns which frames matter.
    Linear head keeps capacity ~= the sklearn logistic baseline (small data)."""
    def __init__(self, d, n_cls):
        super().__init__()
        self.score = nn.Linear(d, 1)
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x, mask):                 # x (B,T,D), mask (B,T) bool
        s = self.score(x).squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return self.head((a * x).sum(1))


class MeanHead(nn.Module):                       # baseline: linear on the mean
    def __init__(self, d, n_cls):
        super().__init__()
        self.head = nn.Linear(d, n_cls)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).float()
        return self.head((x * m).sum(1) / m.sum(1).clamp(min=1))


def _pad(batch_seqs, mu, sd):
    T = max(len(s) for s in batch_seqs)
    B, D = len(batch_seqs), batch_seqs[0].shape[1]
    X = np.zeros((B, T, D), np.float32); M = np.zeros((B, T), bool)
    for i, s in enumerate(batch_seqs):
        X[i, :len(s)] = (s - mu) / sd; M[i, :len(s)] = True
    return torch.tensor(X), torch.tensor(M)


def train_eval(seqs, y, tr, te, classes, model_cls, epochs=200, seed=0):
    torch.manual_seed(seed)
    y2 = np.array([classes.index(v) for v in y])
    allf = np.concatenate([seqs[i] for i in tr], 0)
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    Xtr, Mtr = _pad([seqs[i] for i in tr], mu, sd); ytr = torch.tensor(y2[tr])
    Xte, Mte = _pad([seqs[i] for i in te], mu, sd)
    d = seqs[0].shape[1]
    model = model_cls(d, len(classes)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr, Mtr, ytr = Xtr.to(DEVICE), Mtr.to(DEVICE), ytr.to(DEVICE)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 16):
            b = perm[i:i + 16]
            opt.zero_grad()
            loss = lossf(model(Xtr[b], Mtr[b]), ytr[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(DEVICE), Mte.to(DEVICE)).argmax(1).cpu().numpy()
    return (pred == y2[te]).mean()


def cv(seqs, y, g, model_cls):
    classes = sorted(set(y))
    if len(np.unique(g)) < 2:
        g = np.arange(len(y)) % 4
    accs = []
    for tr, te in GroupKFold(min(3, len(np.unique(g)))).split(seqs, y, g):
        if len(set(y[tr])) < 2 or len(te) < 5:
            continue
        accs.append(train_eval(seqs, y, tr, te, classes, model_cls))
    return float(np.mean(accs)) if accs else np.nan


def main(acts, task, target, group_key, out):
    man = pd.read_parquet(MANIFEST_PATH)
    acts_dir = Path(acts)
    z0 = np.load(next(acts_dir.glob("*.npz")))
    layers = _frame_layers(z0)
    ch = 1 / man[man.task == task].ground_truth.nunique()
    rows = []
    for L in layers:
        seqs, y, g = load_task(acts_dir, man, task, target, group_key, L)
        if seqs is None or len(seqs) == 0:
            continue
        attn = cv(seqs, y, g, AttnPool)
        mean = cv(seqs, y, g, MeanHead)
        rows.append({"layer": L, "attn_acc": round(attn, 3), "mean_acc": round(mean, 3),
                     "chance": round(ch, 3)})
        print(f"  {acts_dir.name} {task} L{L:02d}: attn={attn:.3f}  mean={mean:.3f}  (chance {ch:.3f})")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dst = out / f"probe_seq__{acts_dir.name}__{task}__{target}.csv"
    pd.DataFrame(rows).to_csv(dst, index=False)
    print(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--target", default="ground_truth")
    ap.add_argument("--group-key", default="soundfont")
    ap.add_argument("--out", default="results/trackB/probes")
    a = ap.parse_args()
    main(a.acts, a.task, a.target, a.group_key, a.out)

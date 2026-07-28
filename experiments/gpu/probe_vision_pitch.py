"""Capacity probe: can Qwen2.5-Omni's VISION tower represent pitch — and does an
F0-contour chart beat a spectrogram? Decides whether Track-D's "make it use the
image" problem is READABILITY (fix the image) or USAGE (force attention).

For each shortlist task and each image type (spectrogram | f0contour), we push
the image through model.thinker.visual, mean-pool the merged image tokens (what
the LM actually sees), and linear-probe those features:
  cents_discrimination -> direction {higher,lower} (per-|delta| psychometric curve)
  tuning_judgment      -> signed detune (ridge; MAE vs the 12-TET-snap baseline)
  octave_id            -> octave class (coarse-pitch positive control)
Leakage-safe GroupKFold (hold out base-pitch bins / soundfonts) + permutation null.

  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python gpu/probe_vision_pitch.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import EXP_ROOT, MANIFEST_PATH, RESULTS_DIR  # noqa: E402
from musicprobe.spectrograms import spectrogram_path  # noqa: E402
from musicprobe.f0_contour import f0_contour_path  # noqa: E402

MODEL = "Qwen/Qwen2.5-Omni-7B"
DELTAS = [5, 10, 25, 50, 100]
N_SPLITS = 5
RNG = np.random.default_rng(0)
IMG = {"spectrogram": spectrogram_path, "f0contour": f0_contour_path}


def _facts(f):
    return f if isinstance(f, dict) else json.loads(f)


def load_vision():
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype="auto", device_map="cuda")
    vis = model.thinker.visual
    dev, dtype = model.device, model.dtype
    cache = {}

    def feat(image_rel: str) -> np.ndarray:
        if image_rel in cache:
            return cache[image_rel]
        conv = [{"role": "user", "content": [
            {"type": "image", "image": str(EXP_ROOT / image_rel)},
            {"type": "text", "text": "."}]}]
        inp = proc.apply_chat_template(conv, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt")
        pv = inp["pixel_values"].to(dev, dtype)
        grid = inp["image_grid_thw"].to(dev)
        with torch.no_grad():
            out = vis(pv, grid_thw=grid)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        h = h.reshape(-1, h.shape[-1]).float().mean(0).cpu().numpy()
        cache[image_rel] = h
        return h

    return feat


# ---- probe helpers ----
def _cv_bacc(X, y, g, shuffle=False):
    yy = RNG.permutation(y) if shuffle else y
    accs = []
    for tr, te in GroupKFold(min(N_SPLITS, len(np.unique(g)))).split(X, yy, g):
        if len(np.unique(yy[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], yy[tr])
        accs.append(balanced_accuracy_score(yy[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else np.nan


def _perm_p(X, y, g, obs, n=200):
    null = np.array([_cv_bacc(X, y, g, shuffle=True) for _ in range(n)])
    return float((1 + np.sum(null >= obs)) / (n + 1)), float(np.nanmean(null))


def _curve(X, y, g, delta):
    per = {d: [] for d in DELTAS}
    for tr, te in GroupKFold(min(N_SPLITS, len(np.unique(g)))).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        for d in DELTAS:
            m = delta[te] == d
            if m.sum() >= 2:
                per[d].append((pred[m] == y[te][m]).mean())
    return {d: (round(float(np.mean(v)), 3) if v else None) for d, v in per.items()}


def probe_cents(feat, man, imgtype):
    sub = man[(man.task == "cents_discrimination") & (man.ground_truth != "same")].reset_index(drop=True)
    fac = sub.factors.apply(_facts)
    X = np.array([feat(IMG[imgtype](p)) for p in sub.audio_path])
    y = (sub.ground_truth == "higher").astype(int).to_numpy()
    delta = fac.apply(lambda d: d["delta_cents"]).to_numpy(int)
    g = np.floor(fac.apply(lambda d: d["base_midi"]).to_numpy(float)).astype(int)
    obs = _cv_bacc(X, y, g); p, nmean = _perm_p(X, y, g, obs)
    return {"task": "cents", "img": imgtype, "metric": "bal_acc", "score": round(obs, 3),
            "null": round(nmean, 3), "p": round(p, 4), **{f"acc_{d}c": v for d, v in _curve(X, y, g, delta).items()}}


def probe_tuning(feat, man, imgtype):
    sub = man[man.task == "tuning_judgment"].reset_index(drop=True)
    fac = sub.factors.apply(_facts)
    X = np.array([feat(IMG[imgtype](p)) for p in sub.audio_path])
    y = fac.apply(lambda d: d["detune_cents"] * d["sign"]).to_numpy(float)
    g = fac.apply(lambda d: d["base_midi"]).to_numpy(int)
    snap = float(np.abs(y).mean())
    maes = []
    for tr, te in GroupKFold(min(N_SPLITS, len(np.unique(g)))).split(X, y, g):
        reg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 12)))
        reg.fit(X[tr], y[tr]); maes.append(np.abs(reg.predict(X[te]) - y[te]).mean())
    mae = float(np.mean(maes))
    return {"task": "tuning", "img": imgtype, "metric": "mae_cents", "score": round(mae, 2),
            "null": round(snap, 2), "p": None, "beats_snap": round(snap - mae, 2)}


def probe_octave(feat, man, imgtype):
    sub = man[man.task == "octave_id"].reset_index(drop=True)
    fac = sub.factors.apply(_facts)
    X = np.array([feat(IMG[imgtype](p)) for p in sub.audio_path])
    y = sub.ground_truth.astype(str).to_numpy()
    g = fac.apply(lambda d: d.get("soundfont", "na")).to_numpy()
    if len(np.unique(g)) < 2:
        g = np.arange(len(y)) % N_SPLITS
    obs = _cv_bacc(X, y, g); p, nmean = _perm_p(X, y, g, obs)
    return {"task": "octave", "img": imgtype, "metric": "bal_acc", "score": round(obs, 3),
            "null": round(nmean, 3), "p": round(p, 4), "chance": round(1 / len(np.unique(y)), 3)}


def main():
    man = pd.read_parquet(MANIFEST_PATH)
    feat = load_vision()
    rows = []
    for imgtype in IMG:
        print(f"\n===== {imgtype} =====")
        for fn in (probe_cents, probe_tuning, probe_octave):
            r = fn(feat, man, imgtype); rows.append(r); print(" ", r)
    out = RESULTS_DIR.parent / "trackB" / "probes" / "vision_pitch_probe.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

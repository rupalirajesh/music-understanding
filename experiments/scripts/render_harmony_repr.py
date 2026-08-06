"""Render the Tracks L/M/N/O/P/Q harmony-cluster front-end images
(musicprobe/harmony_repr.py). All six are computed from the AUDIO via
librosa, never from MIDI/factors ground truth -- same non-leakage discipline
as scripts/render_chromagrams.py.

  python scripts/render_harmony_repr.py --kind chroma_picked
  python scripts/render_harmony_repr.py --kind all --limit 20 --force

CPU-only, deterministic (same category as render_chromagrams.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import pandas as pd

from musicprobe.config import EXP_ROOT, MANIFEST_PATH
from musicprobe.harmony_repr import (chroma_picked_path, chroma_picked_zoom_path,
                                      harmony_line_path, harmony_line_zoom_path,
                                      piano_roll_path, tonnetz_path, chroma_zoom_ref_path)
from musicprobe.l1_baselines import KRUMHANSL_MAJ, KRUMHANSL_MIN

TARGET_SR = 22050
HOP = 512          # Track L / Q -- same hop as Track G's chroma_cqt, for comparability
HOP_ZOOM = 128      # Track M -- 4x finer time resolution
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TOP_K = 3           # peak-picking: how many pitch classes stay "on" per frame


def _load(wav_path: Path, sr=TARGET_SR):
    y, orig_sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=sr)


def _chroma(y, sr, hop):
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)


def _peak_pick(chroma, k=TOP_K, silence_frac=0.02):
    """Binarize: keep only the top-k bins per frame, everything else off.
    Removes the overtone/timbre bleed that makes raw chroma blurry even for
    clean single notes -- k=3 covers up to a triad without keeping noise.
    Silent/near-silent frames are left fully off (found via independent
    review + confirmed by testing: argsort-based top-k has no energy floor,
    so a frame with literally zero energy still gets exactly k bins marked
    "on" from tie-broken argsort noise -- fabricated content on any stimulus
    with lead-in/trailing silence or a rest gap)."""
    out = np.zeros_like(chroma)
    frame_energy = chroma.sum(axis=0)
    active = frame_energy > frame_energy.max() * silence_frac if frame_energy.max() > 0 else \
        np.zeros(chroma.shape[1], dtype=bool)
    idx = np.argsort(chroma, axis=0)[-k:, :]
    for t in np.where(active)[0]:
        out[idx[:, t], t] = 1.0
    return out


def render_chroma_picked(wav_path: Path, out_path: Path, hop: int) -> None:
    y = _load(wav_path)
    chroma = _chroma(y, TARGET_SR, hop)
    picked = _peak_pick(chroma)
    t = librosa.times_like(chroma, sr=TARGET_SR, hop_length=hop)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3) if hop == HOP_ZOOM else (6, 3), dpi=150)
    ax.imshow(picked, aspect="auto", origin="lower", cmap="gray_r",
              extent=[t[0] if len(t) else 0, t[-1] if len(t) else 1, 0, 12],
              vmin=0, vmax=1)
    ax.set_yticks(np.arange(12) + 0.5); ax.set_yticklabels(PC_NAMES, fontsize=7)
    ax.set_xlabel("time (s)"); ax.set_title("peak-picked chroma", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _piptrack_points(y, sr, fmin=80.0, fmax=1200.0, hop=HOP, mag_frac=0.15, max_per_frame=4):
    """Audio-derived candidate pitches per frame -- top magnitude piptrack
    bins above mag_frac of that frame's max, capped at max_per_frame so a
    noisy frame doesn't flood the plot. Returns (times, freqs) point arrays."""
    pitches, mags = librosa.piptrack(y=y, sr=sr, hop_length=hop, fmin=fmin, fmax=fmax)
    times = librosa.times_like(pitches, sr=sr, hop_length=hop)
    ts, fs = [], []
    for frame in range(pitches.shape[1]):
        col_mag = mags[:, frame]
        if col_mag.max() <= 0:
            continue
        thresh = col_mag.max() * mag_frac
        cand = np.where(col_mag >= thresh)[0]
        cand = cand[np.argsort(col_mag[cand])[::-1][:max_per_frame]]
        for bin_idx in cand:
            f = pitches[bin_idx, frame]
            if f > 0:
                ts.append(times[frame]); fs.append(f)
    return np.array(ts), np.array(fs)


def render_harmony_line(wav_path: Path, out_path: Path, zoom: bool) -> None:
    y = _load(wav_path)
    ts, fs = _piptrack_points(y, TARGET_SR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    if len(fs):
        ax.scatter(ts, fs, s=4, c="#1a5276", alpha=0.6)
    ax.set_yscale("log")
    if zoom and len(fs):
        med = np.median(fs)
        ax.set_ylim(med * 2 ** (-7 / 12), med * 2 ** (7 / 12))  # +-7 semitones around the action
    else:
        ax.set_ylim(80, 1200)  # fixed axis, not tied to this stimulus -> no leakage
    ax.set_xlabel("time (s)"); ax.set_ylabel("Hz (log)")
    ax.set_title("harmony line (audio-derived pitch candidates)" + (" -- zoomed" if zoom else ""),
                fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _reject_harmonics(freqs_mags: list[tuple[float, float]], tol: float = 0.03) -> list[float]:
    """Greedy fundamental-only filter, same method as
    musicprobe.l1_baselines._fundamentals_in_window: sort candidates by
    frequency ascending, keep a candidate only if it isn't a near-integer
    multiple of an already-accepted (lower) fundamental. Without this,
    piptrack's per-segment magnitude peaks include the harmonic series of a
    single note as if they were separate simultaneous notes (confirmed via
    visual inspection: a 2-note monophonic melodic-interval stimulus
    rendered as 3-4 stacked bars before this filter was added)."""
    cand = sorted({f for f, m in freqs_mags if f > 0})
    funds: list[float] = []
    for f in cand:
        if not any(abs(f / g - round(f / g)) < tol and round(f / g) >= 1 for g in funds):
            funds.append(f)
    return funds


def render_piano_roll(wav_path: Path, out_path: Path) -> None:
    y = _load(wav_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=TARGET_SR, hop_length=HOP, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=TARGET_SR, hop_length=HOP)
    duration = len(y) / TARGET_SR
    bounds = list(onset_times) + [duration]
    if len(bounds) < 2:
        bounds = [0.0, duration]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        seg = y[int(t0 * TARGET_SR):int(t1 * TARGET_SR)]
        if len(seg) < 512:
            continue
        pitches, mags = librosa.piptrack(y=seg, sr=TARGET_SR, hop_length=HOP,
                                         fmin=80.0, fmax=1200.0)
        col_mag = mags.max(axis=1)
        if col_mag.max() <= 0:
            continue
        thresh = col_mag.max() * 0.2
        raw = []
        for bin_idx in np.where(col_mag >= thresh)[0]:
            row = mags[bin_idx, :]
            f = pitches[bin_idx, np.argmax(row)]
            raw.append((f, col_mag[bin_idx]))
        for f in _reject_harmonics(raw)[:4]:  # cap: at most a 4-note chord
            midi = 69 + 12 * np.log2(f / 440)
            ax.barh(midi, t1 - t0, left=t0, height=0.8, color="#1a5276", alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("MIDI note")
    ax.set_ylim(48, 84); ax.set_title("piano-roll (audio-derived onset+pitch)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def render_tonnetz(wav_path: Path, out_path: Path) -> None:
    y = _load(wav_path)
    chroma = _chroma(y, TARGET_SR, HOP)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=TARGET_SR)
    t = librosa.times_like(tonnetz, sr=TARGET_SR, hop_length=HOP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.imshow(tonnetz, aspect="auto", origin="lower", cmap="coolwarm",
              extent=[t[0] if len(t) else 0, t[-1] if len(t) else 1, 0, 6])
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels(["5th-x", "5th-y", "m3-x", "m3-y", "M3-x", "M3-y"], fontsize=7)
    ax.set_xlabel("time (s)"); ax.set_title("tonal centroid (tonnetz)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _estimate_tonic_pc(chroma: np.ndarray) -> int:
    """Krumhansl-profile correlation, same method as
    musicprobe.l1_baselines.key_estimate, restricted to just the tonic
    pitch-class (mode is a nuisance parameter here). Computed from THIS
    stimulus's own chroma -- audio-derived, not ground truth, same
    non-leakage discipline as f0_contour.py's pyin-estimated reference
    pitch for Track D-zoom."""
    profile_sum = chroma.sum(axis=1)
    profile_sum = profile_sum / (profile_sum.sum() + 1e-9)
    best_pc, best_r = 0, -2.0
    for tonic in range(12):
        for profile in (KRUMHANSL_MAJ, KRUMHANSL_MIN):
            r = np.corrcoef(np.roll(profile, tonic), profile_sum)[0, 1]
            if r > best_r:
                best_r, best_pc = r, tonic
    return best_pc


def render_chroma_zoom_ref(wav_path: Path, out_path: Path) -> None:
    """Track X: Track M's zoomed peak-picked chroma, plus the estimated-
    tonic row highlighted and labeled -- the missing zoom+reference
    combination (see musicprobe.harmony_repr.chroma_zoom_ref_path)."""
    y = _load(wav_path)
    chroma = _chroma(y, TARGET_SR, HOP_ZOOM)
    picked = _peak_pick(chroma)
    tonic_pc = _estimate_tonic_pc(chroma)
    t = librosa.times_like(chroma, sr=TARGET_SR, hop_length=HOP_ZOOM)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
    ax.imshow(picked, aspect="auto", origin="lower", cmap="gray_r",
              extent=[t[0] if len(t) else 0, t[-1] if len(t) else 1, 0, 12],
              vmin=0, vmax=1)
    ax.axhspan(tonic_pc, tonic_pc + 1, color="#c0392b", alpha=0.18, zorder=1)
    ax.axhline(tonic_pc, color="#c0392b", lw=1.5, zorder=2)
    ax.axhline(tonic_pc + 1, color="#c0392b", lw=1.5, zorder=2)
    ax.text((t[-1] if len(t) else 1) * 0.99, tonic_pc + 0.5, f" est. tonic: {PC_NAMES[tonic_pc]}",
           color="#c0392b", fontsize=8, va="center", ha="right", zorder=3)
    ax.set_yticks(np.arange(12) + 0.5); ax.set_yticklabels(PC_NAMES, fontsize=7)
    ax.set_xlabel("time (s)"); ax.set_title("zoomed peak-picked chroma + estimated-tonic reference",
                                            fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


RENDERERS = {
    "chroma_picked": lambda w, o: render_chroma_picked(w, o, HOP),
    "chroma_picked_zoom": lambda w, o: render_chroma_picked(w, o, HOP_ZOOM),
    "harmony_line": lambda w, o: render_harmony_line(w, o, zoom=False),
    "harmony_line_zoom": lambda w, o: render_harmony_line(w, o, zoom=True),
    "piano_roll": render_piano_roll,
    "tonnetz": render_tonnetz,
    "chroma_zoom_ref": render_chroma_zoom_ref,
}
PATH_FNS = {
    "chroma_picked": chroma_picked_path, "chroma_picked_zoom": chroma_picked_zoom_path,
    "harmony_line": harmony_line_path, "harmony_line_zoom": harmony_line_zoom_path,
    "piano_roll": piano_roll_path, "tonnetz": tonnetz_path,
    "chroma_zoom_ref": chroma_zoom_ref_path,
}
# harmonic cluster only -- these tasks are what Tracks L-Q target
HARMONY_TASKS = ("key_id", "mode_id", "chord_quality", "interval_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="all", choices=list(RENDERERS) + ["all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--scope", default="battery", choices=["battery", "harmony-only"],
                    help="'battery' (default) renders EVERY stimulus, not just the 4 "
                         "harmony tasks -- required so build_image_jobs' wrong_image "
                         "control (drawn from the whole battery, image_jobs.py) doesn't "
                         "hit a missing-file error. 'harmony-only' is for quick local "
                         "smoke-testing only, not for building a real training manifest.")
    args = ap.parse_args()

    man = pd.read_parquet(MANIFEST_PATH)
    if args.scope == "harmony-only":
        man = man[man.task.isin(HARMONY_TASKS)]
    if args.limit:
        man = man.head(args.limit)

    kinds = list(RENDERERS) if args.kind == "all" else [args.kind]
    for kind in kinds:
        render_fn, path_fn = RENDERERS[kind], PATH_FNS[kind]
        n_done = n_skip = n_err = 0
        for row in man.itertuples():
            out_path = EXP_ROOT / path_fn(row.audio_path)
            if out_path.exists() and not args.force:
                n_skip += 1
                continue
            try:
                render_fn(EXP_ROOT / row.audio_path, out_path)
                n_done += 1
            except Exception as e:
                n_err += 1
                print(f"  [{kind}] ERROR on {row.audio_path}: {e}")
        print(f"[{kind}] rendered {n_done}, skipped {n_skip}, errors {n_err} (of {len(man)} stimuli)")


if __name__ == "__main__":
    main()

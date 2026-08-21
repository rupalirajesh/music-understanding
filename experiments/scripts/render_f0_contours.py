"""Render an F0-CONTOUR line-chart PNG per stimulus — the "readable" image for
Track D's make-the-model-use-the-image experiments (see musicprobe/f0_contour.py).

Pitch is extracted with librosa.pyin (monophonic) and drawn as pitch-over-time
on a LOG-frequency y-axis, so pitch = vertical position — what a chart-reading
vision encoder handles well, unlike a spectrogram texture. Light semitone
gridlines give an absolute reference (the thing the audio path can't recover).

  python scripts/render_f0_contours.py
  python scripts/render_f0_contours.py --limit 20 --force

CPU-only, deterministic (same category as render_spectrograms.py).
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
from musicprobe.f0_contour import f0_contour_path, f0_zoom_path

TARGET_SR = 16000
FMIN, FMAX = 100.0, 1200.0        # covers our E3..E5 stimuli with headroom
YLIM = (90.0, 1300.0)             # fixed axis -> absolute pitch is comparable across images
# per-task half-window (cents) for the ZOOMED image; tuning also gets a bold
# in-tune reference line at the nearest semitone.
ZOOM_SPAN = {"tuning_judgment": 90.0, "cents_discrimination": 160.0}
ZOOM_DEFAULT_SPAN = 450.0


def _extract_f0(wav_path: Path):
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    f0, _, _ = librosa.pyin(y, sr=TARGET_SR, fmin=FMIN, fmax=FMAX,
                            frame_length=2048, hop_length=160)
    t = librosa.times_like(f0, sr=TARGET_SR, hop_length=160)
    return f0, t


def _absolute_label(center_hz: float, has_pitch: bool) -> str:
    """Text label stating the estimated ABSOLUTE pitch -- added 2026-08-21 so
    the same chart that already carries fine relative/cents information also
    carries the one thing it previously discarded by auto-centering. Without
    this, the chart structurally cannot answer any absolute-identification
    question (e.g. PitchBench's 'what MIDI note is this') since the y-axis is
    only ever cents-from-its-own-center, with no absolute anchor printed
    anywhere. This is the ONE addition needed -- same zoom, same recipe,
    consistent across every pitch task, not a per-task special case."""
    if not has_pitch:
        return "center: no clear pitch detected"
    from musicprobe.theory import NOTE_NAMES
    midi = librosa.hz_to_midi(center_hz)
    note_idx = int(round(midi)) % 12
    octave = int(round(midi)) // 12 - 1  # MIDI 60 -> C4, matches l1_baselines convention
    return f"center: {center_hz:.1f} Hz ≈ {NOTE_NAMES[note_idx]}{octave} (MIDI {round(midi)})"


def render_zoom(wav_path: Path, out_path: Path, task: str) -> None:
    """Cents-scale pitch chart, auto-centred so fine differences are visible,
    PLUS a text label stating the absolute pitch the chart is centred on --
    same recipe for every pitch task: estimate -> zoom on the deviation ->
    label the absolute value. Centre + reference are set from the
    pyin-ESTIMATED pitch (not ground truth)."""
    f0, t = _extract_f0(wav_path)
    vf = f0[np.isfinite(f0)]
    span = ZOOM_SPAN.get(task, ZOOM_DEFAULT_SPAN)
    refline = None
    has_pitch = len(vf) > 0
    if not has_pitch:
        # no trackable pitch (e.g. polyphonic note_count) -> still emit a valid,
        # empty chart so downstream image loading never breaks (note_count is the
        # negative control anyway; its image content is irrelevant).
        center = 220.0
        cents = np.full_like(f0, np.nan)
    elif task == "tuning_judgment":
        midi = round(float(librosa.hz_to_midi(np.median(vf))))  # nearest 12-TET note
        center = float(librosa.midi_to_hz(midi))
        refline = 0.0                                           # in-tune = 0 cents
        cents = 1200.0 * np.log2(f0 / center)
    else:
        center = float(np.median(vf))                          # auto-centre on the pitch
        cents = 1200.0 * np.log2(f0 / center)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    for c in range(int(-span), int(span) + 1, 25):             # light 25-cent grid
        ax.axhline(c, color="0.9", lw=0.5, zorder=0)
    if refline is not None:
        ax.axhline(refline, color="#c0392b", lw=1.8, zorder=1)
        ax.text(0.01, refline, " in tune", color="#c0392b", fontsize=9, va="bottom")
    ax.plot(t, cents, color="#0a4fd6", lw=2.6, solid_capstyle="round", zorder=3)
    ax.set_ylim(-span, span)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylabel("pitch (cents from reference)")
    ax.set_xlabel("time (s)")
    ax.set_title(_absolute_label(center, has_pitch), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_one(wav_path: Path, out_path: Path) -> None:
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    f0, voiced, _ = librosa.pyin(y, sr=TARGET_SR, fmin=FMIN, fmax=FMAX,
                                 frame_length=2048, hop_length=160)
    t = librosa.times_like(f0, sr=TARGET_SR, hop_length=160)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    # semitone gridlines from FMIN..FMAX as an absolute pitch reference
    m0, m1 = int(librosa.hz_to_midi(FMIN)), int(librosa.hz_to_midi(FMAX))
    for m in range(m0, m1 + 1):
        ax.axhline(librosa.midi_to_hz(m), color="0.9", lw=0.5, zorder=0)
    ax.plot(t, f0, color="#0a4fd6", lw=2.4, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylabel("pitch (Hz)")
    ax.set_xlabel("time (s)")
    ax.set_yticks([110, 220, 440, 880])
    ax.set_yticklabels(["110", "220", "440", "880"])
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(manifest, exp_root, force, limit, tasks, zoom=False):
    root = Path(exp_root)
    man = pd.read_parquet(manifest)
    if tasks:
        man = man[man.task.isin(tasks)]
    if limit:
        man = man.head(limit)
    path_fn = f0_zoom_path if zoom else f0_contour_path
    done = skipped = failed = 0
    for i, row in enumerate(man.itertuples(), 1):
        out_path = root / path_fn(row.audio_path)
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            if zoom:
                render_zoom(root / row.audio_path, out_path, row.task)
            else:
                render_one(root / row.audio_path, out_path)
            done += 1
        except Exception as e:
            failed += 1
            print(f"!! failed on {row.stimulus_id}: {type(e).__name__}: {e}")
        if i % 100 == 0:
            print(f"{i}/{len(man)} (rendered {done}, skipped {skipped}, failed {failed})")
    print(f"done: rendered {done}, skipped {skipped}, failed {failed}, out of {len(man)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*",
                    default=["octave_id", "tuning_judgment", "cents_discrimination",
                             "note_count", "pitch_note_id"])
    ap.add_argument("--zoom", action="store_true",
                    help="render the ZOOMED cents-scale image (f0zoom/) instead of f0contours/")
    args = ap.parse_args()
    main(args.manifest, args.exp_root, args.force, args.limit, args.tasks, args.zoom)

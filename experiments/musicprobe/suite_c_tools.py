"""Suite C tools -- deterministic, time-resolved measurements the model is
given as a "tool report" (GROUNDING_PILOT_PLAN.md). These are NOT called by
the model; we compute them and hand the output to the model as context.

Fixes the harmony-tool time-resolution gap flagged in the plan: l1_baselines'
key_estimate/chord_quality_estimate collapse a whole clip into one value.
This reuses the same chroma()+Krumhansl-correlation method, unmodified,
just applied per time-window instead of once over the whole clip -- so
`l1_baselines.py` itself is untouched (no risk to its already-reported
numbers).

  python -m musicprobe.suite_c_tools --selftest
"""
import numpy as np

from .l1_baselines import chroma
from .theory import CHORDS, CHORD_SPOKEN, NOTE_NAMES

KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def chord_root_and_quality(c: np.ndarray) -> str:
    """Best-fit (root, quality) chord label, e.g. 'D7', 'G', 'Am' -- unlike
    l1_baselines.chord_quality_estimate, keeps the root (needed to report
    which specific chord, not just its quality)."""
    best, best_r = None, -2
    for root in range(12):
        for name, semis in CHORDS.items():
            template = np.zeros(12)
            for s in semis:
                template[(root + s) % 12] = 1.0
            r = np.corrcoef(template, c)[0, 1]
            if r > best_r:
                best_r, best = r, (root, name)
    root, quality = best
    root_name = NOTE_NAMES[root]
    suffix = {"major": "", "minor": "m", "dominant7": "7", "major7": "maj7",
              "minor7": "m7", "diminished": "dim", "augmented": "aug",
              "sus4": "sus4"}.get(quality, quality)
    return f"{root_name}{suffix}"


def key_root_and_mode(c: np.ndarray) -> str:
    """Best-fit key label, e.g. 'C major', 'A minor' -- same Krumhansl method
    as l1_baselines.key_estimate, factored to take a chroma vector directly
    so it can be called per-window without re-deriving the correlation loop."""
    best, best_r = None, -2
    for tonic in range(12):
        for mode, profile in [("major", KRUMHANSL_MAJ), ("minor", KRUMHANSL_MIN)]:
            r = np.corrcoef(np.roll(profile, tonic), c)[0, 1]
            if r > best_r:
                best_r, best = r, f"{NOTE_NAMES[tonic]} {mode}"
    return best


def windowed_harmony_report(y: np.ndarray, sr: int, window_bounds: list[tuple[float, float]],
                            labels: list[str] | None = None) -> list[dict]:
    """Per-window chord + key estimate -- the actual harmony-tool fix.
    window_bounds: list of (start_sec, end_sec) -- e.g. bar boundaries.
    labels: optional names for each window (e.g. "bar1", "bar2", ...), else
    windows are numbered from 1.
    Returns a list of {"label", "start_s", "end_s", "chord", "key"} dicts --
    this IS the deterministic "tool report" string handed to the model."""
    labels = labels or [f"bar{i+1}" for i in range(len(window_bounds))]
    report = []
    for label, (t0, t1) in zip(labels, window_bounds):
        i0, i1 = int(t0 * sr), int(t1 * sr)
        seg = y[i0:i1]
        if len(seg) < sr * 0.05:  # too short a window to get a reliable chroma read
            report.append({"label": label, "start_s": round(t0, 2), "end_s": round(t1, 2),
                           "chord": None, "key": None})
            continue
        c = chroma(seg, sr)
        report.append({"label": label, "start_s": round(t0, 2), "end_s": round(t1, 2),
                       "chord": chord_root_and_quality(c), "key": key_root_and_mode(c)})
    return report


def format_tool_report(report: list[dict]) -> str:
    """Render the report list as the exact text handed to the model in the
    prompt -- deliberately plain so it's trivial to check whether the
    model's cited measurement_or_tool_output literally matches this string."""
    parts = [f"{r['label']}={r['chord']}" for r in report if r["chord"] is not None]
    return "chord_report: " + ", ".join(parts)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        # Synthetic sanity check: 4 seconds of a clean C-major triad (261.6, 329.6,
        # 392.0 Hz) followed by 4 seconds of a clean G-major triad (392.0, 493.9,
        # 587.3 Hz) -- windowed_harmony_report should read bar1=C, bar2=G.
        sr = 22050
        t = np.linspace(0, 4, sr * 4, endpoint=False)
        c_major = sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0))
        g_major = sum(np.sin(2 * np.pi * f * t) for f in (392.0, 493.9, 587.3))
        y = np.concatenate([c_major, g_major]) * 0.3
        report = windowed_harmony_report(y, sr, [(0, 4), (4, 8)])
        print(format_tool_report(report))
        assert report[0]["chord"] == "C", f"expected bar1=C, got {report[0]['chord']}"
        assert report[1]["chord"] == "G", f"expected bar2=G, got {report[1]['chord']}"
        print("[selftest] PASSED -- windowed_harmony_report correctly distinguishes "
              "C major vs G major across the two windows")
        sys.exit(0)

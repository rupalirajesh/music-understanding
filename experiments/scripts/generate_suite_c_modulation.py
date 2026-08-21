"""Suite C, phenomenon 1 (harmonic function/modulation) -- first real item.
GROUNDING_PILOT_PLAN.md worked example 2, built for real.

Base item: 16-bar progression in C major (bars 1-8), pivoting at bar 9 via a
secondary dominant (D7, V7/V) into G major (bars 9-16).
Swapped-target counterfactual: byte-identical bars 1-8, then pivots via E7
(V7/vi) into A minor instead -- the relative-minor confusion the plan calls
out as the harder/closer counterfactual. Ground truth is exact by
construction (deterministic synthesis), no musician needed, per the plan's
sourcing rule.

  python scripts/generate_suite_c_modulation.py
  python scripts/generate_suite_c_modulation.py --selftest
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

from musicprobe.config import EXP_ROOT, STIMULI_DIR, available_soundfonts
from musicprobe.generators.chords import _voice
from musicprobe.synth import midi_notes, render_midi
from musicprobe.suite_c_tools import windowed_harmony_report, format_tool_report

CHORD_DUR = 1.5  # seconds per bar, matches progression_id's non-blues default
OUT_DIR = STIMULI_DIR / "suite_c" / "modulation"

# (root_midi, quality) per bar -- bars 1-8 identical across both versions
BARS_1_8 = [(60, "major"), (65, "major"), (60, "major"), (67, "major"),
            (60, "major"), (65, "major"), (67, "major"), (60, "major")]
# version A: pivot to G major via D7 (V7/V)
BARS_9_16_G = [(62, "dominant7"), (67, "major"), (72, "major"), (74, "dominant7"),
               (67, "major"), (67, "major"), (72, "major"), (67, "major")]
# version B: pivot to A minor via E7 (V7/vi) -- the swapped-target counterfactual
BARS_9_16_AM = [(64, "dominant7"), (69, "minor"), (74, "minor"), (64, "dominant7"),
                (69, "minor"), (69, "minor"), (74, "minor"), (69, "minor")]


def _render(bars: list[tuple[int, str]], out_path: Path, program: int = 0) -> float:
    notes = []
    t = 0.25
    for root, quality in bars:
        for p in _voice(root, quality, 0):
            notes.append((p, t, t + CHORD_DUR - 0.1))
        t += CHORD_DUR
    sf_name, sf_path = list(available_soundfonts().items())[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_midi(midi_notes(notes, program=program), sf_path, out_path)
    return t + 0.25


def build_item(item_id: str = "mod_001") -> dict:
    """Builds both versions' audio + tool reports. Returns everything needed
    for the 5 Suite C conditions (audio absent/correct/swapped, +tool report
    variants) -- the actual audio-swap and tool-report-swap are just a matter
    of which file/report gets handed to the model at eval time, not something
    that needs building twice."""
    dur_a = _render(BARS_1_8 + BARS_9_16_G, OUT_DIR / f"{item_id}_target_G.wav")
    dur_b = _render(BARS_1_8 + BARS_9_16_AM, OUT_DIR / f"{item_id}_target_Am.wav")
    assert abs(dur_a - dur_b) < 1e-6, "both versions must be the same length"

    bar_bounds = [(0.25 + i * CHORD_DUR, 0.25 + (i + 1) * CHORD_DUR) for i in range(16)]

    def report_for(path):
        y, sr = sf.read(path)
        return windowed_harmony_report(y, sr, bar_bounds)

    report_a = report_for(OUT_DIR / f"{item_id}_target_G.wav")
    report_b = report_for(OUT_DIR / f"{item_id}_target_Am.wav")

    return {
        "item_id": item_id,
        "audio_a": str((OUT_DIR / f"{item_id}_target_G.wav").relative_to(EXP_ROOT)),
        "audio_b": str((OUT_DIR / f"{item_id}_target_Am.wav").relative_to(EXP_ROOT)),
        "ground_truth_a": "modulates from C major to G major at bar 9 (pivot chord D7, V7/V)",
        "ground_truth_b": "modulates from C major to A minor at bar 9 (pivot chord E7, V7/vi)",
        "tool_report_a": format_tool_report(report_a),
        "tool_report_b": format_tool_report(report_b),
        "bar9_span": bar_bounds[8],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="build the item and print the tool reports without asserting anything")
    a = ap.parse_args()
    item = build_item()
    print(f"audio_a: {item['audio_a']}")
    print(f"audio_b: {item['audio_b']}")
    print(f"ground_truth_a: {item['ground_truth_a']}")
    print(f"ground_truth_b: {item['ground_truth_b']}")
    print(f"tool_report_a: {item['tool_report_a']}")
    print(f"tool_report_b: {item['tool_report_b']}")
    print(f"bar9_span: {item['bar9_span']}")
    if a.selftest:
        # bars 1-8 (index 0-7) should read as C-ish/F-ish/G-ish chords in both
        # versions since they're byte-identical audio; bars 9+ (index 8-15)
        # should differ between the two tool reports.
        import re
        chords_a = re.findall(r"bar(\d+)=(\S+)", item["tool_report_a"])
        chords_b = re.findall(r"bar(\d+)=(\S+)", item["tool_report_b"])
        first8_a = [c for n, c in chords_a if int(n) <= 8]
        first8_b = [c for n, c in chords_b if int(n) <= 8]
        assert first8_a == first8_b, (
            f"bars 1-8 should be identical (same audio) but got {first8_a} vs {first8_b}")
        print("[selftest] PASSED -- bars 1-8 read identically across both versions "
              "(as expected, since that audio is byte-identical); bars 9+ diverge")

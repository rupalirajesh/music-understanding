"""GPU entry point for Tracks L-W (the harmony L/M/N/O/P/Q and rhythm
R/S/T/U/V/W representation-ladder sequence, PROJECT_STATE.md next actions
13/14). Registry-based instead of twelve near-duplicate files -- each track
is just (tasks, path_fn, jobs filename, tag); the actual train+eval logic
lives once in gpu/image_track_common.py. See that module's docstring for
why this is a deliberate departure from Track G/H's one-file-per-track
convention (DRY, so a bug fix applies to all twelve at once).

Prereqs (CPU-only, run once on the laptop, already done + committed):
  python scripts/render_harmony_repr.py --kind all     # Tracks L-Q images
  python scripts/render_rhythm_repr.py --kind all      # Tracks R-W images

  python gpu/train_track_repr.py --track L --seed 0 --smoke-test
  python gpu/train_track_repr.py --track L --seed 0
  python gpu/train_track_repr.py --track L --seed 0 --eval-only

Policy (2026-08-05, Rupali's call): run the full L-Q and R-W sequences,
not stop-early -- the goal includes comparing which representation works
best, not just finding any one fix.
"""
import argparse
import sys
from pathlib import Path

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import EXP_ROOT, MANIFEST_DIR  # noqa: E402
from musicprobe.harmony_repr import (chroma_picked_path, chroma_picked_zoom_path,
                                      harmony_line_path, harmony_line_zoom_path,
                                      piano_roll_path, tonnetz_path)  # noqa: E402
from musicprobe.rhythm_repr import (tempogram_path, tempogram_picked_path, onset_line_path,
                                     onset_line_zoom_path, rhythm_roll_path,
                                     rhythm_necklace_path)  # noqa: E402
from image_track_common import run_track  # noqa: E402

HARMONY_TASKS = ("key_id", "mode_id", "chord_quality", "interval_id")
RHYTHM_TASKS = ("tempo_bpm", "beats_per_bar")

TRACKS = {
    "L": dict(name="chroma-picked", tasks=HARMONY_TASKS, path_fn=chroma_picked_path),
    "M": dict(name="chroma-picked-zoom", tasks=HARMONY_TASKS, path_fn=chroma_picked_zoom_path),
    "N": dict(name="harmony-line", tasks=HARMONY_TASKS, path_fn=harmony_line_path),
    "O": dict(name="harmony-line-zoom", tasks=HARMONY_TASKS, path_fn=harmony_line_zoom_path),
    "P": dict(name="piano-roll", tasks=HARMONY_TASKS, path_fn=piano_roll_path),
    "Q": dict(name="tonnetz", tasks=HARMONY_TASKS, path_fn=tonnetz_path),
    "R": dict(name="tempogram", tasks=RHYTHM_TASKS, path_fn=tempogram_path),
    "S": dict(name="tempogram-picked", tasks=RHYTHM_TASKS, path_fn=tempogram_picked_path),
    "T": dict(name="onset-line", tasks=RHYTHM_TASKS, path_fn=onset_line_path),
    "U": dict(name="onset-line-zoom", tasks=RHYTHM_TASKS, path_fn=onset_line_zoom_path),
    "V": dict(name="rhythm-roll", tasks=RHYTHM_TASKS, path_fn=rhythm_roll_path),
    "W": dict(name="rhythm-necklace", tasks=RHYTHM_TASKS, path_fn=rhythm_necklace_path),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(TRACKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    args = ap.parse_args()

    spec = TRACKS[args.track]
    tag = f"qwen25omni-{spec['name']}"
    jobs_path = MANIFEST_DIR / f"{spec['name'].replace('-', '_')}_jobs.parquet"
    ckpt_subdir = f"track_{args.track.lower()}_{spec['name'].replace('-', '_')}_ckpt"

    run_track(seed=args.seed, smoke=args.smoke_test, exp_root=Path(args.exp_root),
              tasks=spec["tasks"], image_path_fn=spec["path_fn"], jobs_path=jobs_path,
              tag=tag, ckpt_subdir=ckpt_subdir, eval_only=args.eval_only)


if __name__ == "__main__":
    main()

"""Track D, Phase 1 — the image hygiene layer (RESEARCH_PLAN.md §12.2).

Additive and separate from jobs.py's build_jobs()/append_jobs() ON PURPOSE:
this is a new experiment on a new input-modality axis, not a v1 battery
change, and jobs.parquet stays completely untouched (PROJECT_STATE.md
decision 10: v1 is frozen). Run:

  python -m musicprobe.image_jobs        # writes manifests/image_jobs.parquet

Each stimulus x format gets THREE jobs, one per image_condition, with audio
held at the correct clip throughout (condition is always "audio" — this
experiment measures what the image ADDS on top of audio, not audio itself):
  image        correct spectrogram (musicprobe.spectrograms.spectrogram_path)
  no_image     no image attached — same job as image, minus the image
  wrong_image  a random spectrogram drawn from anywhere else in the WHOLE
               battery, not filtered to the same task — same "how wrong is
               wrong" design as jobs.py's wrong_audio (jobs.py:8-9, 67-68),
               and same reason: still scored against the ORIGINAL question's
               ground truth, so a model that's actually using the image
               should do worse here, and a model that's ignoring it should
               score the same as the `image` condition. See the 2026-07-24
               report-correction thread for the full reasoning — a same-task
               swap would NOT be a valid control, it has to be a real swap.

Default task scope is the alignment-fixable shortlist (RESEARCH_PLAN.md §12.2
point 5: start where the audio-only probe already says the information is
there, not on the noisier key/mode/chord/interval group).

Renders (scripts/12_render_spectrograms.py) must be run first — this module
reads image paths, it doesn't create them, and will raise clearly if a
required PNG is missing rather than silently building a broken job.
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MANIFEST_DIR, EXP_ROOT
from .manifest import load_manifest
from .prompts import mcq_options, build_prompt
from .spectrograms import spectrogram_path

IMAGE_JOBS_PATH = MANIFEST_DIR / "image_jobs.parquet"

FIXED_CHOICE_TASKS = ("tempo_bpm", "cents_discrimination", "tuning_judgment")
DEFAULT_TASKS = ("octave_id", "tuning_judgment", "cents_discrimination", "note_count")
# image / no_image / wrong_image are the original three (audio always correct).
# image_wrong_audio (correct spectrogram + WRONG audio) is the substitute-vs-
# complement control for the conclusive Track-D run: if the model still scores
# high here, the image is SUBSTITUTING for audio (reading the chart); if it
# collapses, the image genuinely COMPLEMENTS correct audio. Scored against the
# ORIGINAL (correct-audio) question's ground truth, same logic as wrong_image.
IMAGE_CONDITIONS = ("image", "no_image", "wrong_image", "image_wrong_audio")


def _require_rendered(image_path: str) -> None:
    if not (EXP_ROOT / image_path).exists():
        raise FileNotFoundError(
            f"{image_path} doesn't exist — run "
            "scripts/12_render_spectrograms.py before building image jobs.")


def build_image_jobs(tasks: tuple[str, ...] = DEFAULT_TASKS,
                      image_path_fn=spectrogram_path,
                      fixed_choice_tasks: tuple[str, ...] = FIXED_CHOICE_TASKS) -> pd.DataFrame:
    """image_path_fn generalizes this beyond the spectrogram (Track D Phase 1):
    any function audio_path -> image_path works (see chromagram.chromagram_path
    for Track G's harmonic-task front-end) as long as its images are already
    rendered (this module reads paths, it never renders)."""
    man = load_manifest(list(tasks))
    full = load_manifest()                                # partners: full battery
    all_paths = full["audio_path"].tolist()
    all_images = [image_path_fn(p) for p in all_paths]

    rows = []
    for row in man.itertuples():
        fmt = "open" if row.task in fixed_choice_tasks else "mcq"
        correct_image = image_path_fn(row.audio_path)
        _require_rendered(correct_image)

        for image_condition in IMAGE_CONDITIONS:
            r = np.random.default_rng(
                zlib.crc32(f"{row.stimulus_id}|{image_condition}|{fmt}".encode()))
            paraphrase_idx = int(r.integers(3))
            if fmt == "mcq":
                options, answer_letter = mcq_options(row.task, row.ground_truth,
                                                      row.factors, r)
            else:
                options, answer_letter = None, None
            prompt = build_prompt(row.task, paraphrase_idx, fmt, options)

            audio_path = row.audio_path                    # correct audio by default
            if image_condition == "image":
                image_path = correct_image
            elif image_condition == "no_image":
                image_path = None
            elif image_condition == "wrong_image":         # correct audio + wrong image
                image_path = all_images[int(r.integers(len(all_images)))]
                if image_path == correct_image:            # re-draw once on unlucky match
                    image_path = all_images[int(r.integers(len(all_images)))]
                _require_rendered(image_path)
            else:                                          # image_wrong_audio: correct image + WRONG audio
                image_path = correct_image
                j = int(r.integers(len(all_paths)))
                if all_paths[j] == row.audio_path:         # re-draw once on unlucky match
                    j = int(r.integers(len(all_paths)))
                audio_path = all_paths[j]

            rows.append({
                "job_id": f"{row.stimulus_id}::image_{image_condition}::{fmt}",
                "stimulus_id": row.stimulus_id,
                "task": row.task,
                "tier": row.tier,
                # audio correct except in the image_wrong_audio substitute control
                "condition": "wrong_audio" if image_condition == "image_wrong_audio" else "audio",
                "image_condition": image_condition,
                "format": fmt,
                "paraphrase_idx": paraphrase_idx,
                "prompt": prompt,
                "options": "|".join(options) if options else None,
                "answer_letter": answer_letter,
                "ground_truth": row.ground_truth,
                "audio_path": audio_path,
                "image_path": image_path,
            })
    return pd.DataFrame(rows)


def _save(df: pd.DataFrame, out_path: Path = IMAGE_JOBS_PATH) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[image_jobs] {len(df)} jobs across {df['task'].nunique()} tasks -> {out_path} "
          f"({(df.image_condition == 'image').sum()} image, "
          f"{(df.image_condition == 'no_image').sum()} no-image, "
          f"{(df.image_condition == 'wrong_image').sum()} wrong-image)")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spectrogram", choices=["spectrogram", "chromagram"],
                    help="which image_path_fn to build jobs against")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="defaults to DEFAULT_TASKS for spectrogram, the harmonic "
                         "cluster (key_id/mode_id/chord_quality/interval_id) for chromagram")
    ap.add_argument("--out", default=None, help="defaults to manifests/{kind}_jobs.parquet")
    args = ap.parse_args()

    if args.kind == "chromagram":
        from .chromagram import chromagram_path
        path_fn = chromagram_path
        tasks = tuple(args.tasks) if args.tasks else \
            ("key_id", "mode_id", "chord_quality", "interval_id")
        # these 4 tasks are MCQ-primary by design (not in FIXED_CHOICE_TASKS) --
        # no change needed to fixed_choice_tasks for this kind.
        out = Path(args.out) if args.out else MANIFEST_DIR / "chroma_jobs.parquet"
    else:
        path_fn = spectrogram_path
        tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
        out = Path(args.out) if args.out else IMAGE_JOBS_PATH

    _save(build_image_jobs(tasks=tasks, image_path_fn=path_fn), out_path=out)

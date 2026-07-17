"""Expand the stimulus manifest into a table of evaluation *jobs*:
stimulus x prompt-paraphrase x format x condition.

Conditions (the hygiene layer, non-negotiable per MuChoMusic):
  audio        — the real trial
  no_audio     — same question, no audio attached. If a model scores well
                 here, the item measures text priors, not hearing.
  wrong_audio  — audio swapped with a random other stimulus from a different
                 task. Scores above chance mean answer-leakage via the prompt.

Every job carries its full prompt text and answer key, so runners are dumb
executors and everything is reproducible from the jobs file alone.
"""
import zlib

import numpy as np
import pandas as pd

from .config import MANIFEST_DIR, GLOBAL_SEED
from .manifest import load_manifest
from .prompts import mcq_options, build_prompt

JOBS_PATH = MANIFEST_DIR / "jobs.parquet"

NO_AUDIO_FRACTION = 0.30   # fraction of stimuli that also get a no-audio control
WRONG_AUDIO_FRACTION = 0.10
OPEN_FRACTION = 0.25       # fraction that also get an open-ended (non-MCQ) job
EXPLAIN_FRACTION = 0.15    # fraction that also get an answer+explanation job
                           # (not auto-scored; for manual listening-vs-guessing analysis)

FIXED_CHOICE_TASKS = ("tempo_bpm", "cents_discrimination", "tuning_judgment")


def build_jobs(tasks: list[str] | None = None) -> pd.DataFrame:
    man = load_manifest(tasks)
    rng = np.random.default_rng(GLOBAL_SEED)
    jobs = []

    def add(row, condition, fmt, audio_path):
        r = np.random.default_rng(zlib.crc32(f"{row.stimulus_id}|{condition}|{fmt}".encode()))
        paraphrase_idx = int(r.integers(3))
        if fmt == "mcq":
            options, answer_letter = mcq_options(row.task, row.ground_truth,
                                                 row.factors, r)
        else:
            options, answer_letter = None, None
        prompt = build_prompt(row.task, paraphrase_idx, fmt, options)
        jobs.append({
            "job_id": f"{row.stimulus_id}::{condition}::{fmt}",
            "stimulus_id": row.stimulus_id,
            "task": row.task,
            "tier": row.tier,
            "condition": condition,
            "format": fmt,
            "paraphrase_idx": paraphrase_idx,
            "prompt": prompt,
            "options": "|".join(options) if options else None,
            "answer_letter": answer_letter,
            "ground_truth": row.ground_truth,
            "audio_path": audio_path,
        })

    all_paths = man["audio_path"].tolist()
    for row in man.itertuples():
        fmt_main = "open" if row.task in FIXED_CHOICE_TASKS else "mcq"
        add(row, "audio", fmt_main, row.audio_path)
        if rng.random() < NO_AUDIO_FRACTION:
            add(row, "no_audio", fmt_main, None)
        if rng.random() < WRONG_AUDIO_FRACTION:
            other = all_paths[int(rng.integers(len(all_paths)))]
            add(row, "wrong_audio", fmt_main, other)
        if fmt_main == "mcq" and rng.random() < OPEN_FRACTION:
            add(row, "audio", "open", row.audio_path)
        if rng.random() < EXPLAIN_FRACTION:
            add(row, "audio", "explain", row.audio_path)

    df = pd.DataFrame(jobs)
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(JOBS_PATH, index=False)
    print(f"[jobs] {len(df)} jobs "
          f"({(df.condition == 'audio').sum()} audio, "
          f"{(df.condition == 'no_audio').sum()} no-audio, "
          f"{(df.condition == 'wrong_audio').sum()} wrong-audio)")
    return df

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


def _expand(man: pd.DataFrame, all_paths: list[str],
            rng: np.random.Generator) -> list[dict]:
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
    return jobs


def _save(df: pd.DataFrame) -> pd.DataFrame:
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(JOBS_PATH, index=False)
    print(f"[jobs] {len(df)} jobs "
          f"({(df.condition == 'audio').sum()} audio, "
          f"{(df.condition == 'no_audio').sum()} no-audio, "
          f"{(df.condition == 'wrong_audio').sum()} wrong-audio)")
    return df


def build_jobs(tasks: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """FROM-SCRATCH battery build. Refuses to run once any model responses
    exist: rebuilding reshuffles every control draw and wrong-audio pairing,
    which orphans all collected results. Grow the battery with append_jobs()
    instead (see scripts/07_add_instrument_task.py for the pattern)."""
    from .config import RESULTS_DIR
    existing = sorted(RESULTS_DIR.glob("responses__*.parquet"))
    if existing and not force:
        raise RuntimeError(
            f"{len(existing)} response file(s) exist under {RESULTS_DIR} — "
            "rebuilding jobs.parquet would orphan them. Use append_jobs() for "
            "new tasks; build_jobs(force=True) only for a brand-new battery.")
    man = load_manifest(tasks)
    rng = np.random.default_rng(GLOBAL_SEED)
    return _save(pd.DataFrame(_expand(man, man["audio_path"].tolist(), rng)))


def append_jobs(tasks: list[str]) -> pd.DataFrame:
    """Add jobs for newly added tasks WITHOUT rebuilding the rest: existing
    rows pass through byte-identical, so job_ids already answered stay valid
    and reruns remain resumable. (A full build_jobs() would reshuffle the
    wrong-audio pairings of every existing job — never do that after a model
    has run.) Idempotent: already-present job_ids are skipped."""
    old = pd.read_parquet(JOBS_PATH)
    man = load_manifest(tasks)
    rng = np.random.default_rng(GLOBAL_SEED)
    all_paths = load_manifest()["audio_path"].tolist()  # wrong-audio partners: full battery
    new = pd.DataFrame(_expand(man, all_paths, rng))
    new = new[~new["job_id"].isin(set(old["job_id"]))]
    print(f"[jobs] appending {len(new)} new jobs for {tasks}")
    return _save(pd.concat([old, new], ignore_index=True))

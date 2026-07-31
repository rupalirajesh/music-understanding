"""Track H's job hygiene layer (parallel to image_jobs.py, but the axis being
varied is which AUDIO clip is played, not an attached image). Additive and
separate from jobs.py on purpose (PROJECT_STATE.md decision 10: v1 frozen).

  python -m musicprobe.reftone_jobs      # writes manifests/reftone_jobs.parquet

Each tuning_judgment stimulus gets THREE jobs, one per reftone_condition
(ground truth never changes — it's always about whether the TARGET tone,
the second one, is in tune):
  plain          original single-tone audio (the existing tuning_judgment clip)
  reftone        reference tone (correct nominal pitch) + gap + target tone
  wrong_reftone  reference tone shifted a few semitones away + gap + SAME
                 target tone — mechanism control: does a wrong reference
                 mislead the model (real comparison) or not (just reacting
                 to "two tones present")?

Renders (scripts/render_reftones.py) must be run first.
"""
from pathlib import Path

import pandas as pd

from .config import MANIFEST_DIR, EXP_ROOT
from .manifest import load_manifest
from .prompts import build_prompt
from .reftone import reftone_path, wrong_reftone_path

REFTONE_JOBS_PATH = MANIFEST_DIR / "reftone_jobs.parquet"
REFTONE_CONDITIONS = ("plain", "reftone", "wrong_reftone")

_REF_PROMPT_SUFFIX = (
    "\n\nYou will hear TWO tones: a short reference tone first (assume the "
    "reference is exactly in tune), then a pause, then the note to judge. "
    "Decide whether the SECOND tone is in tune or out of tune RELATIVE TO "
    "the reference you just heard.\nAnswer with exactly: in tune, or out of tune."
)


def _require_rendered(path: str) -> None:
    if not (EXP_ROOT / path).exists():
        raise FileNotFoundError(
            f"{path} doesn't exist — run scripts/render_reftones.py first.")


def build_reftone_jobs() -> pd.DataFrame:
    man = load_manifest(["tuning_judgment"])
    rows = []
    for row in man.itertuples():
        ref_path, wrong_path = reftone_path(row.audio_path), wrong_reftone_path(row.audio_path)
        _require_rendered(ref_path); _require_rendered(wrong_path)
        plain_prompt = build_prompt("tuning_judgment", 0, "open", None)
        ref_prompt = plain_prompt.split("\n")[0] + _REF_PROMPT_SUFFIX

        for cond in REFTONE_CONDITIONS:
            audio_path = {"plain": row.audio_path, "reftone": ref_path,
                         "wrong_reftone": wrong_path}[cond]
            prompt = plain_prompt if cond == "plain" else ref_prompt
            rows.append({
                "job_id": f"{row.stimulus_id}::reftone_{cond}::open",
                "stimulus_id": row.stimulus_id,
                "task": row.task,
                "tier": row.tier,
                "condition": "audio",
                "reftone_condition": cond,
                "format": "open",
                "prompt": prompt,
                "ground_truth": row.ground_truth,
                "audio_path": audio_path,
            })
    return pd.DataFrame(rows)


def _save(df: pd.DataFrame) -> pd.DataFrame:
    REFTONE_JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REFTONE_JOBS_PATH, index=False)
    print(f"[reftone_jobs] {len(df)} jobs across {df['task'].nunique()} task(s) -> "
          f"{REFTONE_JOBS_PATH} "
          f"({(df.reftone_condition == 'plain').sum()} plain, "
          f"{(df.reftone_condition == 'reftone').sum()} reftone, "
          f"{(df.reftone_condition == 'wrong_reftone').sum()} wrong-reftone)")
    return df


if __name__ == "__main__":
    _save(build_reftone_jobs())

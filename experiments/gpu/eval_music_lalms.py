"""PROJECT_STATE.md next action 21 -- run MU-LLaMA, MusiLingo, and M2UGen
(via MuMu-LLaMA) against the SAME frozen v1 battery (musicprobe.jobs) already
used for every other Track A model, so results land in the same
responses__<tag>.parquet schema and are directly comparable (same
scoring/analysis pipeline, no separate code path).

Caveat before running this (see PROJECT_STATE next action 21 for the full
reasoning): all three models are adapter+LLM heads on FROZEN MERT embeddings
-- the same encoder already in this project's Track B roster. This doesn't
open a new encoder axis; it tests whether a bigger decoder extracts more
from MERT than the current linear/MLP probes do (next actions 19/13). Worth
sequencing after next action 19 (much cheaper, reuses activations already
on the H100 box) -- if a small MLP head already closes the gap on MERT's
near-floor tasks, these three add less than they'd otherwise appear to.

STATUS 2026-08-12: harness scaffold only, matching the existing
responses__*.parquet schema (run_local.py's convention) -- the three
per-model loaders are STUBS. Unlike Tracks C-Z, which all fine-tune one
already-integrated model family (Qwen2.5-Omni) whose loading code this
project already owns, these are three separate third-party repos this
project has never loaded before -- writing fabricated-looking load code
without the actual repos in hand risks silently-wrong inference calls that
LOOK done but aren't (same trap eval_muchomusic.py's BLOCKER section flags
for its own external dependency). Whoever picks this up on the H100 box:
fill in one _load_* function per model from that repo's own example/demo
script, following the (model, encode) contract below -- everything else
(job iteration, response writing, resumability) is real and ready.

  python gpu/eval_music_lalms.py --model mu-llama
  python gpu/eval_music_lalms.py --model musilingo
  python gpu/eval_music_lalms.py --model m2ugen

Repos (open weights confirmed 2026-08-12 -- see PROJECT_STATE next action 21):
  MU-LLaMA   https://github.com/shansongliu/MU-LLaMA
  MusiLingo  (weights linked from the MU-LLaMA/MusiLingo papers' repos --
             confirm current HF/GitHub location before this run, links move)
  M2UGen     https://github.com/shansongliu/MuMu-LLaMA  (M2UGen was folded
             into this repo under the MuMu-LLaMA name)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import EXP_ROOT, RESULTS_DIR  # noqa: E402
from musicprobe.jobs import JOBS_PATH  # noqa: E402


def _load_mu_llama():
    """TODO (H100 box): clone shansongliu/MU-LLaMA, follow its README to
    load the LLaMA-Adapter-over-MERT checkpoint. Return (model, encode) where
    encode(prompt, audio_path) -> a raw text answer (see contract in run()).
    """
    raise NotImplementedError(
        "MU-LLaMA loader not wired -- see github.com/shansongliu/MU-LLaMA's own "
        "inference script and port its (model, audio) -> text call here.")


def _load_musilingo():
    """TODO (H100 box): MusiLingo pairs frozen MERT embeddings with a linear
    projector into Vicuna. Confirm the current weights location before
    wiring this (link was not re-verified against a primary source
    2026-08-12 -- see module docstring)."""
    raise NotImplementedError(
        "MusiLingo loader not wired -- confirm weights location, then port its "
        "inference call here.")


def _load_m2ugen():
    """TODO (H100 box): clone shansongliu/MuMu-LLaMA (M2UGen), follow its
    README for the music-understanding (not generation) checkpoint/mode."""
    raise NotImplementedError(
        "M2UGen loader not wired -- see github.com/shansongliu/MuMu-LLaMA's own "
        "inference script and port its (model, audio) -> text call here.")


LOADERS = {"mu-llama": _load_mu_llama, "musilingo": _load_musilingo, "m2ugen": _load_m2ugen}


def run(model: str, limit: int | None = None):
    """Same resumable-run pattern as musicprobe/runners/run_local.py: reuse
    the frozen v1 battery (musicprobe.jobs.JOBS_PATH) so these three land in
    the exact schema every other Track A model already uses -- no separate
    scoring/analysis path needed."""
    encode = LOADERS[model]()   # each loader returns a ready-to-call encode(prompt, audio_path)
    jobs = pd.read_parquet(JOBS_PATH)
    out_path = RESULTS_DIR / f"responses__{model}.parquet"
    done = set()
    if out_path.exists():
        done = set(pd.read_parquet(out_path)["job_id"])
    todo = jobs[~jobs["job_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"[{model}] {len(todo)}/{len(jobs)} jobs remaining")
    results = []
    for n, row in enumerate(todo.itertuples(), 1):
        try:
            raw = encode(row.prompt, str(EXP_ROOT / row.audio_path))
            err = None
        except Exception as e:
            raw, err = None, str(e)
        results.append({"job_id": row.job_id, "model": model, "raw_response": raw,
                        "error": err, "ts": time.time()})
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}")
    out = pd.concat([pd.read_parquet(out_path), pd.DataFrame(results)]) if out_path.exists() \
        else pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{model}] wrote {out_path} ({len(out)} total responses)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOADERS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.model, args.limit)

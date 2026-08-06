"""Harness validation: replicate a real published number on a public
benchmark, to sanity-check this project's own scoring pipeline before
trusting its numbers on our novel battery.

Target number (resolved 2026-08-05, primary source arxiv 2408.01337 Table 3):
Qwen-Audio v1 (Qwen/Qwen-Audio-Chat) = 51.4% overall on MuChoMusic. MuChoMusic
does NOT publish a Qwen2-Audio number, so v1 is the only citable target. This
run is a one-off harness sanity check, NOT part of the study's own battery.

------------------------------------------------------------------------------
SCHEMA VERIFIED ON THE BOX 2026-08-06 (`datasets` installed here; the earlier
laptop guesses were wrong). Corrections vs the original stub:
  * split is 'train', NOT 'test'  (there is no test split)
  * there is NO `audio` column and NO `answer_index`
  * MCQ fields: question, correct_answer, distractor_1_answer,
    distractor_2_answer, distractor_3_answer  (+ genre, music_knowledge,
    music_reasoning category lists, num_annotations, odd_question)
  * audio is EXTERNAL, referenced by (dataset, dataset_identifier):
      1187 rows = 329 'sdd' + 858 'musiccaps'
This module now builds the eval-ready MCQ manifest from metadata (done, below)
and STOPS at the two things that need real external work — see BLOCKER.

------------------------------------------------------------------------------
BLOCKER (why this isn't a full run yet) — two external dependencies:

  1. AUDIO. The HF set is metadata-only. Audio must be fetched by
     dataset_identifier from the source corpora:
       - 'sdd'       -> Song Describer Dataset (CC-licensed, Zenodo record
                        10072001 / HF mirror). Cleanly downloadable. 329 rows.
       - 'musiccaps' -> YouTube ids; download via yt-dlp from the MusicCaps
                        csv (google-research musiccaps). Lossy: a large,
                        variable fraction of clips are removed/geo-blocked, so
                        a full 1187-row run comparable to the published 51.4%
                        is NOT reliably reproducible. 858 rows (72%).
     => A clean SDD-only run (329) is possible but is a SUBSET, not comparable
        to the 51.4% overall. Full replication needs the fragile MusicCaps
        scrape. Decision 2026-08-06: DEFER (non-core sanity check; not worth
        the fragile YouTube dependency ahead of the real-music pivot).

  2. MODEL. Qwen-Audio-Chat (v1) is an older model family with its own loader
     (trust_remote_code, different processor API from Qwen2-Audio). `generate`
     below is the remaining stub to implement once audio is in hand.

To finish later: (a) resolve audio paths per source into an `audio_path`
column on the manifest this writes, (b) implement `generate` for
Qwen/Qwen-Audio-Chat, (c) run + compare acc to PUBLISHED_OVERALL.

  python gpu/eval_muchomusic.py --build-manifest   # works now (metadata only)
  python gpu/eval_muchomusic.py                     # raises: audio+model TODO
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_ROOT, MANIFEST_DIR  # noqa: E402

MUCHOMUSIC_HF_ID = "mulab-mir/muchomusic"
QWEN_AUDIO_V1_HF_ID = "Qwen/Qwen-Audio-Chat"   # the model the 51.4% is FOR
PUBLISHED_OVERALL = 0.514                       # Table 3, arxiv 2408.01337
MANIFEST_OUT = MANIFEST_DIR / "muchomusic_jobs.parquet"
LETTERS = ["A", "B", "C", "D"]


def parse_mcq_letter(response: str) -> str | None:
    m = re.search(r"\b([A-D])\b", response.strip()[:20])
    return m.group(1) if m else None


def build_manifest() -> pd.DataFrame:
    """Build the eval-ready MCQ table from metadata (no audio needed).
    Options are shuffled deterministically per question_id; the correct
    letter is recorded. Leaves audio_path empty (external — see BLOCKER)."""
    from datasets import load_dataset
    ds = load_dataset(MUCHOMUSIC_HF_ID, split="train")
    rows = []
    for ex in ds:
        opts = [ex["correct_answer"], ex["distractor_1_answer"],
                ex["distractor_2_answer"], ex["distractor_3_answer"]]
        rng = np.random.default_rng(ex["question_id"])   # deterministic shuffle
        order = rng.permutation(4)
        shuffled = [opts[i] for i in order]
        correct_letter = LETTERS[list(order).index(0)]   # where correct_answer landed
        rows.append({
            "question_id": ex["question_id"],
            "question": ex["question"],
            "options": "|".join(shuffled),
            "correct_letter": correct_letter,
            "audio_source": ex["dataset"],               # 'sdd' | 'musiccaps'
            "audio_identifier": ex["dataset_identifier"],
            "audio_path": None,                           # EXTERNAL — see BLOCKER
        })
    df = pd.DataFrame(rows)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST_OUT, index=False)
    print(f"[muchomusic] built {len(df)} MCQ rows -> {MANIFEST_OUT} "
          f"({(df.audio_source=='sdd').sum()} sdd, "
          f"{(df.audio_source=='musiccaps').sum()} musiccaps; audio NOT attached)")
    return df


def run(smoke_test: bool = False):
    raise NotImplementedError(
        "MuChoMusic full run is DEFERRED (see module docstring BLOCKER): needs "
        "(1) external audio — SDD downloadable, MusicCaps=YouTube/lossy — resolved "
        "onto the manifest's audio_path, and (2) a Qwen-Audio-Chat v1 `generate` "
        "loader. Run `--build-manifest` to (re)build the metadata MCQ table now.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-manifest", action="store_true",
                   help="build the metadata MCQ manifest (works now, no audio/GPU)")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    if args.build_manifest:
        build_manifest()
    else:
        run(smoke_test=args.smoke_test)

"""Harness validation: replicate a real published number on a public
benchmark, to sanity-check this project's own scoring pipeline before
trusting its numbers on our novel battery.

Resolved 2026-08-05 (was blocked on this, PROJECT_STATE next action 5):
checked the MuChoMusic paper's actual results table (arxiv 2408.01337,
Table 3) via primary source, not a secondhand summary. It evaluates
"Qwen-Audio" (v1, arxiv 2311.07919) at 51.4% overall / 51.1% knowledge /
51.0% reasoning / 89.7% instruction-following. It does NOT evaluate
Qwen2-Audio (v2, arxiv 2407.10759) -- no follow-up paper found with a
standalone original-MuChoMusic number for v2 either (checked
arxiv 2504.00369, which only cites v1's number secondhand). So there is
NO published Qwen2-Audio MuChoMusic number to replicate against.

Consequence: point this harness at Qwen-Audio v1 (Qwen/Qwen-Audio-Chat on
HF), not Qwen2-Audio -- v1 is the only model with a citable ground-truth
number for this specific benchmark. Qwen2-Audio stays on the main Track A
roster as-is; this run is purely a harness sanity check, run once, not
part of the study's own battery.

UNVERIFIED, needs the H100 box to actually run: dataset field names below
are my best read of the HF dataset card (mulab-mir/muchomusic), not
confirmed against the real schema -- `datasets`/`huggingface_hub` aren't
installed in the laptop venv, so this has not executed end to end. Smoke-
test on ~20 rows before a full run, same discipline as every gpu/ script.

  python gpu/eval_muchomusic.py --smoke-test
  python gpu/eval_muchomusic.py
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_ROOT  # noqa: E402

MUCHOMUSIC_HF_ID = "mulab-mir/muchomusic"
QWEN_AUDIO_V1_HF_ID = "Qwen/Qwen-Audio-Chat"  # the model the 51.4% is FOR
PUBLISHED_OVERALL = 0.514  # Table 3, arxiv 2408.01337 -- Qwen-Audio v1 only

LETTERS = ["A", "B", "C", "D"]


def parse_mcq_letter(response: str) -> str | None:
    m = re.search(r"\b([A-D])\b", response.strip()[:20])
    return m.group(1) if m else None


def run(smoke_test: bool = False) -> pd.DataFrame:
    from datasets import load_dataset  # deferred: not in laptop venv
    from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401

    ds = load_dataset(MUCHOMUSIC_HF_ID, split="test")
    if smoke_test:
        ds = ds.select(range(min(20, len(ds))))

    # TODO verify against the real schema once `datasets` is installed:
    # expected columns roughly {audio, question, choices, answer_index,
    # category (knowledge/reasoning)} per the paper's description -- adjust
    # field names below to match ds.features before a real run.
    model_id = QWEN_AUDIO_V1_HF_ID
    print(f"Loading {model_id} ...")
    # model, processor = load once we're on the H100 box; left as a stub
    # since this can't be exercised on the laptop.

    rows = []
    for ex in ds:
        # response = generate(model, processor, ex["audio"], ex["question"], ex["choices"])
        response = None  # placeholder until run on GPU
        parsed = parse_mcq_letter(response) if response else None
        correct_letter = LETTERS[ex["answer_index"]] if "answer_index" in ex else None
        rows.append({
            "id": ex.get("id"),
            "category": ex.get("category"),
            "raw_response": response,
            "parsed": parsed,
            "correct": (parsed == correct_letter) if parsed and correct_letter else None,
        })

    df = pd.DataFrame(rows)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULTS_ROOT / "muchomusic_qwenaudio_v1.parquet"
    df.to_parquet(out, index=False)
    acc = df["correct"].mean()
    print(f"n={len(df)}  accuracy={acc:.3f}  published={PUBLISHED_OVERALL:.3f}  "
          f"delta={acc - PUBLISHED_OVERALL:+.3f}")
    print(f"wrote {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    run(smoke_test=args.smoke_test)

"""Export EVERYTHING about a model's run to human-readable files for manual
verification — no information loss, nothing pre-digested.

  .venv/bin/python scripts/04_export_for_review.py --model gemini-2.5-flash

Writes to results/review__<model>/:
  <task>.csv          every job: full prompt, options, ground truth, the model's
                      COMPLETE verbatim response, parsed answer, correct flag,
                      condition, audio path (so you can listen alongside)
  explanations.csv    all explain-format responses across tasks, verbatim —
                      the listening-vs-guessing evidence file
  summary.csv         the accuracy table (for orientation, not as the verdict)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musicprobe.config import RESULTS_DIR
from musicprobe.scoring import score_model, summary_table

COLS = ["job_id", "stimulus_id", "task", "condition", "format",
        "paraphrase_idx", "audio_path", "prompt", "options", "ground_truth",
        "raw_response", "parsed", "correct", "error", "factors"]


def main(model: str):
    df = score_model(model)
    df = df.copy()
    df["factors"] = df["factors"].apply(json.dumps)
    out_dir = RESULTS_DIR / f"review__{model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task, g in df.groupby("task"):
        g[COLS].sort_values(["condition", "stimulus_id"]).to_csv(
            out_dir / f"{task}.csv", index=False)

    exp = df[df["format"] == "explain"]
    if len(exp):
        exp[COLS].sort_values(["task", "stimulus_id"]).to_csv(
            out_dir / "explanations.csv", index=False)

    summary_table(df).to_csv(out_dir / "summary.csv", index=False)
    print(f"wrote {out_dir}/ — {len(df)} rows across "
          f"{df['task'].nunique()} task files + explanations.csv + summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    main(args.model)

"""Bundle every model's scored run into one Excel workbook for eyeball analysis.

  .venv/bin/python scripts/06_export_excel.py

Writes results/analysis_workbook.xlsx:
  audio / no_audio / wrong_audio   per model x task: n, n_correct, accuracy,
                                   refusals — split into MCQ vs open columns
                                   (mcq >> open = MCQ-inflation check)
  data__<model>                    every job with every column: identifiers,
                                   audio path, full prompt, options, ground
                                   truth, verbatim response, parsed, correct —
                                   filter/sort in Excel, cross-check by ear
  explain__<model>                 explain-format responses verbatim (not
                                   auto-scored; the listening-vs-guessing file)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musicprobe.config import RESULTS_DIR
from musicprobe.scoring import REFUSAL_RE

DATA_COLS = ["job_id", "stimulus_id", "task", "tier", "condition", "format",
             "paraphrase_idx", "audio_path", "prompt", "options",
             "ground_truth", "raw_response", "parsed", "correct", "refused",
             "error", "factors"]
CONDITIONS = ["audio", "no_audio", "wrong_audio"]


def condition_sheet(df: pd.DataFrame, cond: str) -> pd.DataFrame:
    g = df[(df["condition"] == cond) & (df["format"] != "explain")]
    rows = []
    for (model, task), gg in g.groupby(["model", "task"], sort=False):
        rec = {"model": model, "task": task, "tier": gg["tier"].iloc[0],
               "n_total": len(gg), "n_correct": int(gg["correct"].eq(True).sum()),
               "accuracy": gg["correct"].eq(True).mean(),
               "n_refused": int(gg["refused"].sum())}
        for fmt, gf in gg.groupby("format"):
            rec[f"n_{fmt}"] = len(gf)
            rec[f"n_correct_{fmt}"] = int(gf["correct"].eq(True).sum())
            rec[f"acc_{fmt}"] = gf["correct"].eq(True).mean()
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "tier", "task"])


def main():
    frames = []
    for p in sorted(RESULTS_DIR.glob("scored__*.parquet")):
        df = pd.read_parquet(p)
        df["refused"] = (df["correct"].isna()
                         & df["raw_response"].fillna("").str.contains(REFUSAL_RE))
        frames.append(df)
    if not frames:
        sys.exit("no scored__*.parquet in results/ — run scoring first")
    allof = pd.concat(frames, ignore_index=True)
    allof = allof[allof["model"] != "dry"]

    out = RESULTS_DIR / "analysis_workbook.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for cond in CONDITIONS:
            condition_sheet(allof, cond).to_excel(xl, sheet_name=cond, index=False)
        for model, g in allof.groupby("model", sort=False):
            tag = model.replace("/", "_")[-22:]  # sheet names cap at 31 chars
            g[DATA_COLS].sort_values(["task", "condition", "stimulus_id"]) \
                .to_excel(xl, sheet_name=f"data__{tag}", index=False)
            exp = g[g["format"] == "explain"]
            if len(exp):
                exp[DATA_COLS].sort_values(["task", "stimulus_id"]) \
                    .to_excel(xl, sheet_name=f"explain__{tag}", index=False)
    print(f"wrote {out} — {allof['model'].nunique()} models, {len(allof)} rows")


if __name__ == "__main__":
    main()

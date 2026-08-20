"""Mentor's ask (2026-08-19): before treating "run our fine-tuned model across a
benchmark suite" as a contribution, check whether the model we're fine-tuning
(Qwen2.5-Omni-7B) already does well on the benchmarks in our own portfolio
(BENCHMARK_LANDSCAPE.md Sec5). This answers it for MUSE Benchmark specifically,
at zero GPU cost, by parsing logs the MUSE authors already generated and
committed to their own public repo -- not a new inference run.

MUSE Benchmark (github.com/brandoncarone/MUSE_music_benchmark, arxiv 2510.19055)
ran Qwen2.5-Omni-7B themselves and logged every question/response/verdict under
`Gemini_Qwen_AF_logs/`. Their own paper reports this qualitatively ("at or near
chance on advanced tasks"); this script re-derives it quantitatively, per task,
with the actual chance level for each task's answer format (checked directly
against each log's `expected=` field, not assumed).

Requires a local clone of the MUSE repo (NOT vendored here -- their audio
license (LICENSE_DATA.md) permits non-commercial research use but not
redistribution; the logs are text-only and already public in their repo, so
this script reads them in place rather than copying them into this repo):

  git clone https://github.com/brandoncarone/MUSE_music_benchmark.git <path>
  python gpu/parse_musebench_qwen.py --muse-dir <path>                    # baseline (default)

FINE-TUNED COMPARISON (added 2026-08-19, same session as the baseline pull):
`gpu/patch_musebench_lora.py` generates LoRA-wrapped copies of MUSE's own 10
runner scripts, tagged e.g. "LORA-e_f0text" in their log filenames so they
never collide with the original baseline logs. Once those patched scripts
have been run (see that script's docstring) and produced real logs in some
directory, point this script at them with `--log-dir`/`--model-suffix`:

  python gpu/parse_musebench_qwen.py --muse-dir <path> \\
      --log-dir <wherever the patched-script run wrote its .log files> \\
      --model-suffix Qwen2.5-Omni-LORA-e_f0text \\
      --out results/external_benchmarks/muse_qwen25omni-lora-e_f0text.csv

Then diff that CSV's `acc` column against `muse_qwen25omni.csv`'s (the
baseline, already committed) for the real controlled delta -- same 10 tasks,
same prompts/stimuli/scoring, only the model weights differ.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_ROOT  # noqa: E402

DEFAULT_MODEL_SUFFIX = "Qwen2.5-Omni"

TIER = {
    "instrumentID": "beginner", "contourID": "beginner", "oddballs": "beginner",
    "rhythm_matching": "beginner", "transposition": "beginner",
    "chord_quality": "advanced", "key_modulation": "advanced",
    "chord_progression_matching": "advanced", "syncopation": "advanced",
    "meter_identification": "advanced",
}

# Chance level per task, read off the *union* of every `expected=` value seen
# across all logs for that task (verified directly, not assumed from the
# README's task descriptions) -- see the module docstring's discipline note.
CHANCE = {
    "instrumentID": None,  # >2 instrument classes, varies by trial; not a flat chance level
    "contourID": 0.25,     # 4-way: Arch / Ascending / Descending / (4th shape, e.g. Plateau)
    "oddballs": 0.50,      # binary Yes/No (is there an oddball)
    "rhythm_matching": 0.50,   # binary Yes/No (same rhythm)
    "transposition": 0.50,     # binary Yes/No (is this a transposition)
    "chord_quality": 0.50,     # binary Major/Minor
    "key_modulation": 0.50,    # binary Yes/No (did the key modulate)
    "chord_progression_matching": 0.50,  # binary Yes/No (same progression)
    "syncopation": 0.50,       # binary A/B (which excerpt is more syncopated)
    "meter_identification": 0.33,  # >=3-way (Groups of 3/4/5[/more]); conservative (best-case) chance
}


def parse(muse_dir: Path, log_dir: Path | None = None, model_suffix: str = DEFAULT_MODEL_SUFFIX) -> pd.DataFrame:
    log_dir = log_dir or (muse_dir / "Gemini_Qwen_AF_logs")
    if not log_dir.exists():
        raise FileNotFoundError(f"{log_dir} not found -- clone the MUSE repo first (see module "
                                 "docstring), or pass --log-dir for a fine-tuned run's logs")
    log_re = re.compile(rf"(.+?)_{re.escape(model_suffix)}_CHAT_(COT|SYSINST)_Group([AB])_seed(\d)\.log")

    rows = []
    for path in sorted(log_dir.glob(f"*_{model_suffix}_*.log")):
        m = log_re.match(path.name)
        if not m:
            continue
        task, mode, group, seed = m.groups()
        text = path.read_text(errors="replace")
        correct = len(re.findall(r"Evaluation: Correct", text))
        incorrect = len(re.findall(r"Evaluation: Incorrect", text))
        rows.append(dict(task=task, tier=TIER[task], mode=mode, group=group,
                          seed=int(seed), correct=correct, total=correct + incorrect))
    if not rows:
        raise RuntimeError(f"no {model_suffix} logs matched under {log_dir}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["task", "tier"], as_index=False).agg(correct=("correct", "sum"), n=("total", "sum"))
    agg["acc"] = agg["correct"] / agg["n"]
    agg["chance"] = agg["task"].map(CHANCE)
    agg["at_or_below_chance"] = agg.apply(
        lambda r: (r["acc"] <= r["chance"] + 0.05) if pd.notna(r["chance"]) else None, axis=1)
    return agg.sort_values(["tier", "task"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse-dir", required=True, type=Path,
                     help="path to a local clone of brandoncarone/MUSE_music_benchmark")
    ap.add_argument("--log-dir", type=Path, default=None,
                     help="default: <muse-dir>/Gemini_Qwen_AF_logs (the baseline logs already "
                          "in the repo). Point this at a fine-tuned run's own log output dir instead.")
    ap.add_argument("--model-suffix", default=DEFAULT_MODEL_SUFFIX,
                     help="the '...' in '<task>_<suffix>_CHAT_...log' -- default 'Qwen2.5-Omni' "
                          "(baseline); use e.g. 'Qwen2.5-Omni-LORA-e_f0text' for a "
                          "patch_musebench_lora.py fine-tuned run")
    ap.add_argument("--out", type=Path,
                     default=RESULTS_ROOT / "external_benchmarks" / "muse_qwen25omni.csv")
    args = ap.parse_args()

    df = parse(args.muse_dir, log_dir=args.log_dir, model_suffix=args.model_suffix)
    summary = summarize(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {args.out}")

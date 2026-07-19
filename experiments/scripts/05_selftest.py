"""Pipeline self-test — run BEFORE burning GPU/API budget on a real model.

  .venv/bin/python scripts/05_selftest.py

Checks, in order:
  1. Manifest/file integrity: every audio job's WAV exists and loads; every
     no-audio job carries no path (the NaN-path regression).
  2. Runner loop on a fake backend that CRASHES on bad paths — proves the
     NaN fix in the actual runner code path, not a copy of it.
  3. Resume/retry: errored jobs are retried on rerun, successes are kept.
  4. Oracle scoring: feed the known-correct answer for every job through the
     real parser/scorer -> accuracy must be ~1.0 for every task (catches
     parser/option/answer-key mismatches for every task and format).
  5. Review export runs and contains verbatim responses.
Exits non-zero on any failure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import soundfile as sf

from musicprobe.config import EXP_ROOT, RESULTS_DIR
from musicprobe.jobs import JOBS_PATH
from musicprobe import scoring
from musicprobe.runners import run_api

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


print("1. file integrity")
jobs = pd.read_parquet(JOBS_PATH)
missing, unreadable = [], []
for p in jobs.loc[jobs["condition"] != "no_audio", "audio_path"].unique():
    f = EXP_ROOT / p
    if not f.exists():
        missing.append(p)
    else:
        try:
            y, sr = sf.read(f)
            if len(y) == 0:
                unreadable.append(p)
        except Exception:
            unreadable.append(p)
check("all referenced WAVs exist", not missing, f"{len(missing)} missing")
check("all WAVs readable and non-empty", not unreadable, f"{len(unreadable)} bad")
na = jobs[jobs["condition"] == "no_audio"]["audio_path"]
check("no-audio jobs carry no path", na.isna().all())
check("every job has prompt text", jobs["prompt"].str.len().gt(10).all())
mcq = jobs[jobs["format"] == "mcq"]
check("every MCQ job has options + answer key",
      mcq["options"].notna().all() and mcq["answer_letter"].notna().all())
ok_key = all(o.split("|")[ord(a) - 65] == t for o, a, t in
             zip(mcq["options"], mcq["answer_letter"], mcq["ground_truth"]))
check("answer_letter indexes the ground truth option", ok_key)

print("2. runner handles NaN paths (strict backend)")
def strict_backend(prompt, audio_path, model, max_tokens=64):
    if audio_path is not None:
        assert isinstance(audio_path, str), f"non-string path: {audio_path!r}"
        assert (EXP_ROOT / audio_path).exists()
    return "A"
run_api.get_backend = lambda model: strict_backend  # route through the real loop
out = run_api.run("selftest-strict", limit=None)
res = pd.read_parquet(out)
check("strict backend: zero errors over all jobs",
      res["error"].isna().all(), f"{res['error'].notna().sum()} errors")

print("2b. run_local loop handles NaN paths (strict fake loader)")
from musicprobe.runners import run_local
def strict_generate(prompt, audio_path, max_new_tokens):
    if audio_path is not None:
        assert isinstance(audio_path, str), f"non-string path: {audio_path!r}"
        assert (EXP_ROOT / audio_path).exists()
    return "A"
run_local.pick_loader = lambda model: (lambda m: strict_generate)  # factory contract
out_l = run_local.run("selftest-local")
res_l = pd.read_parquet(out_l)
check("run_local: zero errors over all jobs",
      res_l["error"].isna().all() and len(res_l) == len(jobs),
      f"{res_l['error'].notna().sum()} errors")

print("3. resume/retry logic")
res.loc[res.index[:25], "error"] = "SimulatedError"
res.loc[res.index[:25], "raw_response"] = None
res.to_parquet(out, index=False)
run_api.run("selftest-strict")
res2 = pd.read_parquet(out)
check("errored jobs retried, all clean after rerun",
      res2["error"].isna().all() and len(res2) == len(jobs))
check("no duplicate job_ids after resume", not res2["job_id"].duplicated().any())

print("4. oracle scoring (correct answers through the real parser)")
def oracle(job):
    if job.format == "mcq":
        return f"{job.answer_letter}."
    if job.format == "explain":
        return f"{job.ground_truth}. I heard it at 0.5s."
    return str(job.ground_truth)
oracle_rows = [{"job_id": j.job_id, "model": "selftest-oracle",
                "raw_response": oracle(j), "error": None, "ts": 0.0}
               for j in jobs.itertuples()]
pd.DataFrame(oracle_rows).to_parquet(
    RESULTS_DIR / "responses__selftest-oracle.parquet", index=False)
df = scoring.score_model("selftest-oracle")
tab = scoring.summary_table(df)
bad = tab[(tab["acc_audio"] < 0.999) | (tab["unparseable_audio"] > 0.001)]
check("oracle scores ~100% on every task", len(bad) == 0,
      f"tasks off: {bad['task'].tolist()}" if len(bad) else "")
psy = scoring.psychometric_cents(df)
check("cents psychometric table populated", len(psy) >= 5 and psy["n"].sum() > 100)
tun = scoring.psychometric_tuning(df)
check("tuning psychometric table populated", len(tun) >= 5)

print("5. review export")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rev", Path(__file__).parent / "04_export_for_review.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main("selftest-oracle")
out_dir = RESULTS_DIR / "review__selftest-oracle"
files = list(out_dir.glob("*.csv"))
check("per-task CSVs + explanations + summary written",
      len(files) >= jobs["task"].nunique() + 2, f"{len(files)} files")
exp = pd.read_csv(out_dir / "explanations.csv")
check("explanations.csv holds verbatim responses",
      exp["raw_response"].str.contains("I heard").all())

# cleanup selftest artifacts
for f in RESULTS_DIR.glob("*selftest*"):
    if f.is_file():
        f.unlink()
import shutil
shutil.rmtree(out_dir, ignore_errors=True)

print()
if FAILURES:
    print(f"SELFTEST FAILED: {FAILURES}")
    sys.exit(1)
print("SELFTEST PASSED — pipeline is safe to run on real models")

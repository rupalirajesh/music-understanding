"""Score model responses against ground truth and produce the analysis
artifacts: per-task accuracy (audio vs no-audio vs wrong-audio), confusion
matrices, tempo error distribution, and the cents psychometric curve.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .jobs import JOBS_PATH
from .manifest import load_manifest
from .prompts import LETTERS


# ------------------------------------------------------------------ parsing

# Declining to answer ("I cannot hear audio") is epistemically honest on
# no-audio controls — it must count as incorrect (not be dropped from the
# denominator, which biases acc toward the rows where the model guessed),
# and be reported separately from genuine parse failures.
REFUSAL_RE = re.compile(
    r"cannot|can'?t|unable|sorry|please provide|need (?:the|to (?:see|hear))"
    r"|text-based|don'?t have", re.I)

def parse_response(row) -> str | None:
    """Extract the model's answer as a canonical string, or None if unparseable."""
    raw = (row.raw_response or "").strip()
    if not raw:
        return None
    if row.format == "explain":
        return None  # never auto-scored; read verbatim in the review export
    if row.task == "tuning_judgment":
        low = raw.lower()
        if "out of tune" in low or "off" in low:
            return "out of tune"
        if "in tune" in low:
            return "in tune"
        return None
    if row.format == "mcq":
        m = re.match(r"^\s*\(?([A-Da-d])\)?[.):\s]", raw + " ")
        if m:
            letter = m.group(1).upper()
            opts = row.options.split("|")
            return opts[LETTERS.index(letter)] if LETTERS.index(letter) < len(opts) else None
        # fall through: maybe the model answered with the option text itself
        low = raw.lower()
        for o in row.options.split("|"):
            if o.lower() in low:
                return o
        return None
    if row.task == "instrument_id":  # open format; MCQ handled above
        low = raw.lower()
        for w in ("piano", "violin", "flute", "synth"):
            if w in low:
                return "synth lead" if w == "synth" else w
        return None
    if row.task == "tempo_bpm":
        m = re.search(r"\d+(?:\.\d+)?", raw)
        return m.group(0) if m else None
    if row.task == "cents_discrimination":
        low = raw.lower()
        for w in ("higher", "lower", "same"):
            if w in low:
                return w
        return None
    return raw  # open-ended: scored with substring match below (or LLM judge later)


def is_correct(task: str, parsed: str | None, truth: str) -> bool | None:
    if parsed is None or (not isinstance(parsed, str) and pd.isna(parsed)):
        return None
    if task == "tempo_bpm":
        try:
            est, true = float(parsed), float(truth)
        except ValueError:
            return None
        return abs(est - true) / true <= 0.08  # within 8% counts as correct
    return parsed.strip().lower() == truth.strip().lower() or \
        truth.strip().lower() in parsed.strip().lower()


# ------------------------------------------------------------------ scoring

def score_model(model: str) -> pd.DataFrame:
    resp_path = RESULTS_DIR / f"responses__{model.replace('/', '_')}.parquet"
    resp = pd.read_parquet(resp_path)
    jobs = pd.read_parquet(JOBS_PATH)
    df = jobs.merge(resp[["job_id", "raw_response", "error"]], on="job_id")
    df["parsed"] = [parse_response(r) for r in df.itertuples()]
    df["correct"] = [is_correct(r.task, r.parsed, r.ground_truth)
                     for r in df.itertuples()]
    man = load_manifest()[["stimulus_id", "factors"]]
    df = df.merge(man, on="stimulus_id", how="left")
    df["model"] = model
    out = RESULTS_DIR / f"scored__{model.replace('/', '_')}.parquet"
    save = df.copy()
    save["factors"] = save["factors"].apply(json.dumps)
    save.to_parquet(out, index=False)
    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per task: accuracy under each condition + the text-prior verdict.
    Accuracy is over ALL jobs in the condition — refusals and parse failures
    score as incorrect, and are reported separately (refused_* / unparseable_*).
    Explain-format jobs are excluded (manual analysis only)."""
    df = df[df["format"] != "explain"]
    rows = []
    for task, g in df.groupby("task"):
        rec = {"task": task, "tier": g["tier"].iloc[0], "n_audio": 0}
        for cond, gg in g.groupby("condition"):
            unanswered = gg["correct"].isna()
            refused = unanswered & gg["raw_response"].fillna("").str.contains(REFUSAL_RE)
            rec[f"acc_{cond}"] = gg["correct"].eq(True).mean()
            rec[f"refused_{cond}"] = refused.mean()
            rec[f"unparseable_{cond}"] = (unanswered & ~refused).mean()
            if cond == "audio":
                rec["n_audio"] = int(len(gg))
        a, na = rec.get("acc_audio", np.nan), rec.get("acc_no_audio", np.nan)
        if not np.isnan(a) and not np.isnan(na):
            rec["audio_gain"] = a - na  # <~0.05 => task measures text priors, flag it
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["tier", "task"]).reset_index(drop=True)


def confusion(df: pd.DataFrame, task: str) -> pd.DataFrame:
    g = df[(df["task"] == task) & (df["condition"] == "audio")
           & df["parsed"].notna()]
    return pd.crosstab(g["ground_truth"], g["parsed"],
                       rownames=["truth"], colnames=["answered"])


def psychometric_cents(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy vs |delta_cents| for the discrimination task (3AFC chance = 1/3)."""
    g = df[(df["task"] == "cents_discrimination")
           & (df["condition"] == "audio")].copy()
    g["delta"] = g["factors"].apply(lambda f: f["delta_cents"])
    tab = g.groupby("delta")["correct"].agg(["mean", "count"]).reset_index()
    tab.columns = ["delta_cents", "accuracy", "n"]
    return tab


def psychometric_tuning(df: pd.DataFrame) -> pd.DataFrame:
    """Detection rate of 'out of tune' vs detune size (12-TET grid probe).
    A representation snapped to 12-TET predicts a flat curve near 0."""
    g = df[(df["task"] == "tuning_judgment") & (df["condition"] == "audio")
           & (df["format"] != "explain")].copy()
    g["detune"] = g["factors"].apply(lambda f: f["detune_cents"])
    g["said_out"] = g["parsed"] == "out of tune"
    tab = g.groupby("detune").agg(said_out_rate=("said_out", "mean"),
                                  n=("said_out", "count")).reset_index()
    return tab


def tempo_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Log2 ratio of estimate to truth: peaks at ±1 are octave errors (honest
    listening); a spike at fixed BPMs regardless of truth means priors."""
    g = df[(df["task"] == "tempo_bpm") & (df["condition"] == "audio")
           & df["parsed"].notna()].copy()
    g["true_bpm"] = g["ground_truth"].astype(float)
    g["est_bpm"] = pd.to_numeric(g["parsed"], errors="coerce")
    g = g.dropna(subset=["est_bpm"])
    g["log2_ratio"] = np.log2(g["est_bpm"] / g["true_bpm"])
    return g[["stimulus_id", "true_bpm", "est_bpm", "log2_ratio"]]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    df = score_model(args.model)
    print(summary_table(df).to_string(index=False))
    print("\n--- cents psychometric ---")
    print(psychometric_cents(df).to_string(index=False))

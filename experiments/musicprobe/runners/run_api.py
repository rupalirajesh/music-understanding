"""Run eval jobs against API models (black-box Track A / L3).

Usage (from experiments/):
  .venv/bin/python -m musicprobe.runners.run_api --model dry
  .venv/bin/python -m musicprobe.runners.run_api --model gemini-2.5-flash --limit 50
  .venv/bin/python -m musicprobe.runners.run_api --model gpt-4o-audio-preview

Keys: GEMINI_API_KEY / OPENAI_API_KEY env vars.
Resumable: already-answered job_ids are skipped, results appended to
results/responses__<model>.parquet after every batch. Temperature 0 always.
"""
import argparse
import base64
import time
from pathlib import Path

import pandas as pd

from ..config import RESULTS_DIR, EXP_ROOT
from ..jobs import JOBS_PATH


def _wav_bytes(rel_path: str) -> bytes:
    return (EXP_ROOT / rel_path).read_bytes()


# ------------------------------------------------------------------ backends

def call_dry(prompt: str, audio_path: str | None, model: str, max_tokens: int = 64) -> str:
    return "A"  # plumbing test: constant answer; accuracy should equal position rate


def call_openai(prompt: str, audio_path: str | None, model: str, max_tokens: int = 64) -> str:
    from openai import OpenAI  # lazy import
    client = OpenAI()
    content = [{"type": "text", "text": prompt}]
    if audio_path:
        b64 = base64.b64encode(_wav_bytes(audio_path)).decode()
        content.append({"type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"}})
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}])
    return resp.choices[0].message.content or ""


def call_gemini(prompt: str, audio_path: str | None, model: str, max_tokens: int = 64) -> str:
    from google import genai  # lazy import
    from google.genai import types
    client = genai.Client()
    parts = [types.Part.from_text(text=prompt)]
    if audio_path:
        parts.append(types.Part.from_bytes(data=_wav_bytes(audio_path),
                                           mime_type="audio/wav"))
    resp = client.models.generate_content(
        model=model, contents=parts,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=max_tokens))
    return resp.text or ""


def get_backend(model: str):
    if model == "dry":
        return call_dry
    if model.startswith(("gpt-", "o")):
        return call_openai
    if model.startswith("gemini"):
        return call_gemini
    raise ValueError(f"don't know which API serves model '{model}'")


# ------------------------------------------------------------------ main loop

def run(model: str, limit: int | None = None, tasks: list[str] | None = None):
    jobs = pd.read_parquet(JOBS_PATH)
    if tasks:
        jobs = jobs[jobs["task"].isin(tasks)]
    out_path = RESULTS_DIR / f"responses__{model.replace('/', '_')}.parquet"
    done: set = set()
    results = []
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        prev = prev[prev["error"].isna()]  # errored jobs get retried on rerun
        done = set(prev["job_id"])
        results = prev.to_dict("records")
    todo = jobs[~jobs["job_id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"[run] model={model}: {len(todo)} jobs to run ({len(done)} already done)")

    backend = get_backend(model)
    for n, row in enumerate(todo.itertuples(), 1):
        try:
            mt = 512 if row.format == "explain" else 64
            raw = backend(row.prompt, row.audio_path, model, mt)
            err = None
        except Exception as e:  # log and continue; rerun picks failures up again
            raw, err = None, f"{type(e).__name__}: {e}"
            time.sleep(2)
        results.append({
            "job_id": row.job_id, "model": model, "raw_response": raw,
            "error": err, "ts": time.time(),
        })
        if n % 20 == 0 or n == len(todo):
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            ok = [r for r in results if r.get("error") is None]
            pd.DataFrame(results).to_parquet(out_path, index=False)
            print(f"  {n}/{len(todo)} done ({len(results) - len(ok)} errors)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="dry | gpt-4o-audio-preview | gemini-2.5-flash | ...")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args()
    run(args.model, args.limit, args.tasks)

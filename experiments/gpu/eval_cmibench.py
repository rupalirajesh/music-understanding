"""CMI-Bench first-party run for Qwen2.5-Omni-7B (mentor's ask 2026-08-19 —
see BENCHMARK_LANDSCAPE.md Sec6 and PROJECT_STATE.md next action 26).

CMI-Bench (github.com/nicolaus625/CMI-bench, arxiv 2506.12285) has never
evaluated Qwen2.5-Omni -- only Qwen-Audio v1 and Qwen2-Audio-Instruct are on
its own leaderboard, unlike MUSE Benchmark where an existing Qwen2.5-Omni log
could just be parsed (see parse_musebench_qwen.py). This is a real first-party
inference run, not a re-derivation.

SCHEMA VERIFIED against the real repo 2026-08-19 (cloned, read directly, not
guessed -- same discipline as eval_muchomusic.py's schema correction):
  - Input: `data/<Source>/CMI_<task>.jsonl`, one JSON object per line with
    `instruction`, `output` (ground truth), `audio_path` (list, usually
    length 1), `audio_start`/`audio_end` (seconds, `end=-1` means "to EOF"),
    `split` (use only `split[0] == "test"`).
  - Audio is NOT pre-cropped -- these are windows into full-length source
    recordings, so cropping must happen at load time. Reuses CMI-Bench's own
    `data_loader.load_audio(path, target_sr, start, end)` for this (their own
    crop+resample logic, already correctness-tested against their own scoring
    pipeline) rather than re-deriving crop math here.
  - **Output must be a JSON ARRAY** (`json.dump(results, f)`), NOT JSONL --
    their own `evaluate.py` does `data = json.load(f)`, not
    `[json.loads(l) for l in f]`. Each element:
    `{"question": ..., "response": <model's raw text>, "correct_answer": ...,
    "audioid": ..., "other": ""}` -- field names copied verbatim from
    `model/infer.py`'s own `results.append(...)`, not invented.
  - Written to `{cmi_dir}/model/results/{model}/{model}_<task>.jsonl` (the
    extension says jsonl but the content is a JSON array -- that mismatch is
    THEIRS, preserved here so `evaluate.py`'s hardcoded glob
    `model/results/{model}/{model}*.jsonl` finds it unmodified).

Model loading mirrors `musicprobe/runners/run_local.py`'s already-verified
`load_qwen_omni` (disable_talker(), dtype-cast floating inputs,
return_audio=False) -- reimplemented here rather than imported, because that
function loads audio itself from an EXP_ROOT-relative path with no cropping,
which doesn't fit CMI-Bench's "window into a full-length recording" audio.
This version takes a pre-loaded, pre-cropped waveform instead.

SETUP (external, not vendored into this repo -- same reasoning as
parse_musebench_qwen.py; CC BY 4.0 per the CMI-Bench repo, but there's no
reason to duplicate a multi-GB third-party corpus into our own history):
  git clone https://github.com/nicolaus625/CMI-bench.git <cmi_dir>
  cd <cmi_dir> && wget https://huggingface.co/datasets/nicolaus625/CMI-bench/resolve/main/test_Data.zip
  unzip test_Data.zip -d test   # -> <cmi_dir>/test/data/<Source>/... matching each row's audio_path

RUN (H100 box -- needs a GPU, not runnable on the laptop):
  python gpu/eval_cmibench.py --cmi-bench-dir <cmi_dir> --limit 5   # smoke test first, eyeball responses
  python gpu/eval_cmibench.py --cmi-bench-dir <cmi_dir>             # full run, resumable per task file
  cd <cmi_dir> && python evaluate.py --model qwen25omni --task all # their own scorer, unmodified

NOTE for whoever runs this: `evaluate.py --model` uses an `argparse` `choices=`
allowlist that does not include "qwen25omni" -- either add it there (one-line
edit, safe: their own scoring functions are keyed off the task name in the
filename, not the model name) or symlink/copy this run's output under an
allowed model tag (e.g. "qwen2") before scoring. Flagging rather than
silently picking one, since it changes what the score files are named.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

MODEL_TAG = "qwen25omni"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"

# Tasks this project cares about for the mentor's per-domain question
# (BENCHMARK_LANDSCAPE.md Sec6 table) -- pitch/key/melody/beat/technique, not
# the captioning/tagging tasks already flagged as contamination-risky (Sec2).
# Pass --task-glob 'CMI_*.jsonl' to run everything instead.
PRIORITY_TASKS = [
    "CMI_GS_key.jsonl", "CMI_Nsynth_pitch.jsonl", "CMI_VocalSet_tech.jsonl",
    "CMI_Guzheng_Tech.jsonl", "CMI_gtzan_beat.jsonl", "CMI_ballroom_beat.jsonl",
    "CMI_MedleyDB.jsonl",
]


def load_model(model_name: str, lora_checkpoint: str | None = None):
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto").eval()
    # NOT calling disable_talker() -- confirmed 2026-08-21: on the installed
    # transformers version, disable_talker() deletes self.talker, but
    # generate()'s talker_kwargs dict unconditionally reads
    # self.talker.codec_pad_token regardless of return_audio, crashing any
    # later generate() call. Costs ~10GB extra; fits an 80GB A100 fine.
    if lora_checkpoint:
        # same wrapping pattern as image_track_common.load_for_eval /
        # attention_audio.py's --lora-checkpoint: the LoRA adapter only ever
        # targets .thinker (the LLM decoder), never the audio tower -- see
        # RESEARCH_PLAN.md Sec12.6 / next action 9.
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora_checkpoint)
        print(f"[eval_cmibench] wrapped .thinker with adapter from {lora_checkpoint}")
    return processor, model


def generate(processor, model, prompt: str, audio) -> str:
    """audio: 1-D numpy waveform at 16kHz."""
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": "stimulus.wav"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio], sampling_rate=16000,
                        return_tensors="pt", padding=True).to(model.device)
    for k in list(inputs.keys()):
        if torch.is_floating_point(inputs[k]):
            inputs[k] = inputs[k].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def run(cmi_dir: Path, model_name: str, output_dir: Path, task_names: list[str], limit: int | None,
        lora_checkpoint: str | None = None, model_tag: str = MODEL_TAG):
    sys.path.insert(0, str(cmi_dir))
    from data_loader import load_audio  # CMI-Bench's own crop/resample logic

    processor, model = load_model(model_name, lora_checkpoint)
    out_root = output_dir / model_tag
    out_root.mkdir(parents=True, exist_ok=True)

    task_files = sorted(p for p in (cmi_dir / "data").glob("*/*.jsonl") if p.name in task_names)
    missing = set(task_names) - {p.name for p in task_files}
    if missing:
        print(f"WARNING: not found under {cmi_dir/'data'}: {sorted(missing)}")

    for jsonl_path in task_files:
        out_path = out_root / f"{model_tag}_{jsonl_path.name[4:]}"  # strip 'CMI_' prefix, matches their own convention
        if out_path.exists():
            print(f"skip (exists): {out_path.name}")
            continue
        rows = [json.loads(l) for l in jsonl_path.read_text().splitlines()]
        rows = [r for r in rows if r["split"][0] == "test"]
        if limit:
            rows = rows[:limit]
        print(f"{jsonl_path.parent.name}/{jsonl_path.name}: {len(rows)} test rows")

        results = []
        skipped_multi_audio = 0
        for row in rows:
            if len(row["audio_path"]) != 1:
                skipped_multi_audio += 1
                continue  # comparison-style prompts with 2+ clips -- not in PRIORITY_TASKS; skip rather than guess a multi-audio prompt format
            audio_path = cmi_dir / "test" / row["audio_path"][0].lstrip("./")
            wav = load_audio(str(audio_path), target_sr=16000,
                              start=row.get("audio_start", 0.0),
                              end=row.get("audio_end", 30.0))
            wav = wav.squeeze(0).numpy()
            response = generate(processor, model, row["instruction"], wav)
            results.append({
                "question": row["instruction"],
                "response": response,
                "correct_answer": row["output"],
                "audioid": str(audio_path),
                "other": "",
            })
        if skipped_multi_audio:
            print(f"  skipped {skipped_multi_audio} multi-audio rows")

        # NOTE: JSON array, not JSONL, despite the .jsonl extension -- matches
        # model/infer.py's own `json.dump(results, f, indent=4)` exactly, see
        # module docstring. evaluate.py reads it back with `json.load(f)`.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"wrote {out_path} ({len(results)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmi-bench-dir", required=True, type=Path)
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--output-dir", type=Path, default=None,
                     help="default: <cmi-bench-dir>/model/results, matching their own convention so evaluate.py finds it unmodified")
    ap.add_argument("--task-glob", default=None,
                     help="e.g. 'CMI_*.jsonl' for everything; default is PRIORITY_TASKS (the mentor's-question-relevant subset)")
    ap.add_argument("--limit", type=int, default=None, help="rows per task, for a smoke test")
    ap.add_argument("--lora-checkpoint", default=None,
                     help="path to a saved PEFT adapter dir (e.g. one of Track C/E/D-zoom's "
                          "checkpoints) to wrap .thinker with, for a baseline-vs-fine-tuned "
                          "comparison on real audio -- see BENCHMARK_LANDSCAPE.md Sec6 / "
                          "PROJECT_STATE.md next action 26")
    ap.add_argument("--model-tag", default=None,
                     help="output subfolder/filename tag; default 'qwen25omni' for baseline, "
                          "or 'qwen25omni-<lora-checkpoint dirname>' when --lora-checkpoint is set "
                          "-- keeps baseline and fine-tuned runs from overwriting each other")
    args = ap.parse_args()
    output_dir = args.output_dir or (args.cmi_bench_dir / "model" / "results")
    if args.task_glob:
        names = [p.name for p in (args.cmi_bench_dir / "data").glob(f"*/{args.task_glob}")]
    else:
        names = PRIORITY_TASKS
    tag = args.model_tag or (f"{MODEL_TAG}-{Path(args.lora_checkpoint).name}" if args.lora_checkpoint else MODEL_TAG)
    run(args.cmi_bench_dir, args.model_name, output_dir, names, args.limit,
        lora_checkpoint=args.lora_checkpoint, model_tag=tag)

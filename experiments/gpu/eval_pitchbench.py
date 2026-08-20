"""PitchBench first-party run for Qwen2.5-Omni-7B (mentor's ask 2026-08-19 --
see BENCHMARK_LANDSCAPE.md Sec6 and PROJECT_STATE.md next action 26).
PitchBench (arxiv 2605.26176, "Measuring Pitch Hearing in Audio-Language
Models") is this project's own §5 pick as the "narrow-task baseline" --
synthetic, controlled pitch psychophysics, closest published neighbor to our
own battery's L1/L2/L3 pitch tasks (`pitch_note_id`/`cents_discrimination`/
`octave_id`). It evaluated 6 frontier models (Gemini 3.1 Pro/3 Flash, GPT-4o
audio, Qwen-3.5 Omni Plus/Flash, Audio Flamingo Next Instruct) -- NOT
Qwen2.5-Omni, so unlike MUSE there's no existing log to mine; this is a real
first-party run, same status as CMI-Bench.

SCHEMA VERIFIED 2026-08-19 via HF's datasets-server API directly (not the
paper text, not a secondhand summary -- same "primary source" discipline as
eval_muchomusic.py's schema correction):
  - HF dataset `pitchbench-authors/PitchBench`, CC BY 4.0, no gating.
  - 30 configs (`pitchbench_a1_single_pitch_id` ... `pitchbench_f2_melodic
    _line_tonal`), all with a single "test" split. Confirmed via
    https://datasets-server.huggingface.co/splits?dataset=pitchbench-authors/PitchBench
  - Each row: `audio` (wav), `source` (instrument), and FOUR parallel
    ground-truth/prompt pairs for the same clip: `gt_midi`/`prompt_midi`
    (int 0-127), `gt_abc`/`prompt_abc` (Scientific Pitch Notation),
    `gt_solfege`/`prompt_solfege`, `gt_freq`/`prompt_freq` (Hz). This script
    always uses the `_midi` pair (cleanest to parse/score) unless overridden.
  - NOT every config's rows necessarily carry the exact same 4-field
    single-pitch shape (categories B/C/D/E/F test timestamps, chords,
    sequences, robustness) -- only verified category A's schema above
    directly. `--config` lets you point this at one config to eyeball its
    actual fields before trusting the generic path; `run()` fails loudly
    (KeyError) rather than silently mis-scoring a config whose schema
    differs, so a real run will surface any config this script doesn't
    handle rather than fabricate a number for it.

Model call mirrors `eval_cmibench.py`'s (itself mirroring the already-run
`musicprobe/runners/run_local.py::load_qwen_omni`) -- reimplemented rather
than imported because PitchBench delivers audio as in-memory HF `Audio`
arrays, not local file paths, so there's no path to hand run_local.py's
loader.

SETUP: none needed beyond `pip install datasets soundfile` -- audio streams
from HF directly, nothing to vendor (unlike CMI-Bench/MUSE's separate
download step).

RUN (H100 box):
  python gpu/eval_pitchbench.py --config pitchbench_a1_single_pitch_id --limit 5  # smoke test one config
  python gpu/eval_pitchbench.py                                                   # all 30 configs, resumable
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

MODEL_TAG = "qwen25omni"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
HF_DATASET = "pitchbench-authors/PitchBench"

ALL_CONFIGS = [
    "pitchbench_a1_single_pitch_id", "pitchbench_a2_single_pitch_by_loudness",
    "pitchbench_a3_single_pitch_by_duration", "pitchbench_b1_single_pitch_within_silence",
    "pitchbench_b2_pitch_at_timestamp", "pitchbench_b3_timestamp_single_pitch",
    "pitchbench_b4_timestamp_specific_pitch", "pitchbench_b5_timestamp_multiple_pitches",
    "pitchbench_c1_chord_count_pitches", "pitchbench_c2_chord_dyad_interval",
    "pitchbench_c3_chord_quality", "pitchbench_c4_chord_pitches",
    "pitchbench_d1_sequence_count_pitches", "pitchbench_d2_dyad_lower_higher_difference",
    "pitchbench_d3_contour_discrete", "pitchbench_d4_contour_continuous",
    "pitchbench_d5_sequence_ranking_by_pitch", "pitchbench_d6_sequence_dyad_interval",
    "pitchbench_d7a_pitch_with_reference", "pitchbench_d7b_pitch_with_reference_split",
    "pitchbench_d8_sequence_pitches", "pitchbench_e1_audio_effects", "pitchbench_e2_background",
    "pitchbench_e3_harmonic_saturation", "pitchbench_e4_time_stretching", "pitchbench_e5_vibrato",
    "pitchbench_e6_slightly_off", "pitchbench_f1_melodic_line_atonal", "pitchbench_f2_melodic_line_tonal",
]

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_ROOT  # noqa: E402


def load_model(model_name: str, lora_checkpoint: str | None = None):
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto").eval()
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    if lora_checkpoint:
        # same .thinker-only wrapping as eval_cmibench.py / image_track_common.py
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora_checkpoint)
        print(f"[eval_pitchbench] wrapped .thinker with adapter from {lora_checkpoint}")
    return processor, model


def generate(processor, model, prompt: str, audio) -> str:
    """audio: 1-D numpy waveform, sample rate given by `sr`."""
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
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def parse_midi(response: str):
    m = re.search(r"-?\d+", response)
    return int(m.group()) if m else None


def run(model_name: str, out_dir: Path, configs: list[str], limit: int | None,
        lora_checkpoint: str | None = None, model_tag: str = MODEL_TAG):
    from datasets import load_dataset
    import librosa

    processor, model = load_model(model_name, lora_checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        out_path = out_dir / f"{model_tag}_{config}.json"
        if out_path.exists():
            print(f"skip (exists): {out_path.name}")
            continue
        ds = load_dataset(HF_DATASET, config, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        print(f"{config}: {len(ds)} rows, fields={ds.column_names}")

        results = []
        for row in ds:
            prompt = row["prompt_midi"]  # see module docstring: fails loudly (KeyError) if a config's schema differs
            gt = row["gt_midi"]
            audio = row["audio"]  # HF Audio feature: {'array': np.ndarray, 'sampling_rate': int}
            wav = librosa.resample(audio["array"], orig_sr=audio["sampling_rate"], target_sr=16000) \
                if audio["sampling_rate"] != 16000 else audio["array"]
            response = generate(processor, model, prompt, wav)
            pred = parse_midi(response)
            results.append({
                "config": config, "prompt": prompt, "response": response,
                "gt_midi": gt, "pred_midi": pred,
                "correct": (pred == gt) if pred is not None else False,
                "source": row.get("source"),
            })

        n_correct = sum(r["correct"] for r in results)
        print(f"  acc={n_correct}/{len(results)} = {n_correct/len(results):.3f}" if results else "  (empty)")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--config", default=None, help="one config name; default runs all 30")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "external_benchmarks" / "pitchbench_raw")
    ap.add_argument("--limit", type=int, default=None, help="rows per config, for a smoke test")
    ap.add_argument("--lora-checkpoint", default=None,
                     help="path to a saved PEFT adapter dir -- see BENCHMARK_LANDSCAPE.md Sec6")
    ap.add_argument("--model-tag", default=None,
                     help="default 'qwen25omni' for baseline, or derived from the checkpoint dirname")
    args = ap.parse_args()
    configs = [args.config] if args.config else ALL_CONFIGS
    tag = args.model_tag or (f"{MODEL_TAG}-{Path(args.lora_checkpoint).name}" if args.lora_checkpoint else MODEL_TAG)
    run(args.model_name, args.out_dir, configs, args.limit,
        lora_checkpoint=args.lora_checkpoint, model_tag=tag)

"""BASS first-party run for Qwen2.5-Omni-7B (mentor's ask 2026-08-19 -- see
BENCHMARK_LANDSCAPE.md Sec6 and PROJECT_STATE.md next action 26). BASS
(arxiv 2602.04085, github.com/minjang10/bass_music_benchmark) is this
project's "supplementary" §5 pick, not the anchor -- its "musicological
analysis" category is gene/attribute-dominance identification, not
key/chord/harmony content despite the name (already noted in §5); its real
value here is real-audio scale (1,993 songs) and structural/artist-reasoning
tasks nothing else in the portfolio covers.

STATUS 2026-08-19: dataset schema confirmed live (HF `oreva/bass_music_
benchmark`, 4 configs: `structural_segmentation`, `lyrics_transcription`,
`artist_collaboration`, `musicological_analysis`, all a single "test" split,
checked via datasets-server), model-call code below is real and mirrors the
already-verified pattern in eval_cmibench.py/eval_pitchbench.py -- but
**audio resolution is an open blocker, not solved here**, same category of
issue as eval_muchomusic.py's own BLOCKER section, flagged rather than
guessed around:
  - Each row carries BOTH an `audio` field (a bare filename, e.g.
    "collab_analysis_2031.wav" -- not a path, not embedded bytes) AND a
    `youtube_url` list. Checked via datasets-server's first-rows endpoint
    directly, not the paper or README.
  - The BASS GitHub repo's own `run_evaluation.py` (the file this script's
    interface mirrors) reads `question['audio_path']` as if it already
    resolves to a loadable file -- implying the intended workflow downloads
    the real audio separately (their own paper's 1,993 songs are real
    commercial-adjacent tracks by artist/collaboration, same copyright
    category as this project's own shelved famous-song pilot,
    `BENCHMARK_LANDSCAPE.md` §5) and places it locally keyed by the `audio`
    filename -- NOT verified where that download step lives (no obvious
    audio distribution found alongside the HF metadata during this pass).
  - Until that's resolved, this script builds the eval-ready manifest per
    category (real, from the real dataset) and stops at the same place
    eval_muchomusic.py stopped for its own audio-resolution blocker, rather
    than fabricate a download path that might be wrong.

RUN (audio resolution still needed first -- see BLOCKER above):
  python gpu/eval_bass.py --build-manifest --category artist_collaboration
  python gpu/eval_bass.py --category artist_collaboration --audio-dir <resolved audio dir>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

MODEL_TAG = "qwen25omni"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
HF_DATASET = "oreva/bass_music_benchmark"
CATEGORIES = ["structural_segmentation", "lyrics_transcription",
              "artist_collaboration", "musicological_analysis"]
MCQ_TASKS = {"Single-Gene Detection", "Pairwise-Gene Detection", "Gene Dominance Ranking"}

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR.parent))
from musicprobe.config import RESULTS_ROOT  # noqa: E402


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
        from peft import PeftModel
        model.thinker = PeftModel.from_pretrained(model.thinker, lora_checkpoint)
        print(f"[eval_bass] wrapped .thinker with adapter from {lora_checkpoint}")
    return processor, model


def generate(processor, model, prompt: str, audio, sr: int = 16000) -> str:
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio": "stimulus.wav"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=[audio], sampling_rate=sr,
                        return_tensors="pt", padding=True).to(model.device)
    for k in list(inputs.keys()):
        if torch.is_floating_point(inputs[k]):
            inputs[k] = inputs[k].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def build_manifest(category: str, out_path: Path):
    """Real dataset, no audio needed -- confirms rows/fields/prompts are what
    this script expects, same "verify before trusting" step as every other
    real_music_*.py manifest builder in this project."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, category, split="test")
    print(f"{category}: {len(ds)} rows, fields={ds.column_names}")
    rows = [dict(r) for r in ds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_path} (audio NOT resolved -- see module docstring BLOCKER)")


def run(category: str, audio_dir: Path, model_name: str, out_dir: Path, limit: int | None,
        lora_checkpoint: str | None = None, model_tag: str = MODEL_TAG):
    import random
    import soundfile as sf
    import librosa
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, category, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    processor, model = load_model(model_name, lora_checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_tag}_{category}_output.json"

    outputs = []
    missing_audio = 0
    for row in ds:
        audio_path = audio_dir / row["audio"]
        if not audio_path.exists():
            missing_audio += 1
            continue
        y, sr = sf.read(str(audio_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)

        if row["task"] in MCQ_TASKS:
            choices = [f"{ac}: {desc}" for ac, desc in row["answer_choices_with_descriptions"].items()]
            preds = []
            for _ in range(4):  # BASS's own protocol: 4 shuffles per MCQ row, see README/run_evaluation.py
                random.shuffle(choices)
                q = row["question"] + "\nAnswer choices:\n" + "\n".join(choices)
                preds.append(generate(processor, model, q, y))
            prediction = preds
        else:
            prediction = generate(processor, model, row["question"], y)

        output = dict(row)
        output["prediction"] = prediction
        outputs.append(output)

    if missing_audio:
        print(f"WARNING: {missing_audio}/{len(ds)} rows had no local audio at {audio_dir} -- see BLOCKER in module docstring")
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"wrote {out_path} ({len(outputs)}/{len(ds)} rows scored)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=CATEGORIES)
    ap.add_argument("--build-manifest", action="store_true",
                     help="just pull the real dataset + confirm schema, no model/audio needed")
    ap.add_argument("--audio-dir", type=Path, default=None,
                     help="local dir containing the audio referenced by each row's bare `audio` filename -- resolving this is the open BLOCKER, see module docstring")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "external_benchmarks" / "bass_raw")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lora-checkpoint", default=None,
                     help="path to a saved PEFT adapter dir -- see BENCHMARK_LANDSCAPE.md Sec6")
    ap.add_argument("--model-tag", default=None,
                     help="default 'qwen25omni' for baseline, or derived from the checkpoint dirname")
    args = ap.parse_args()
    if args.build_manifest:
        build_manifest(args.category, args.out_dir / f"{args.category}_manifest.json")
    else:
        if args.audio_dir is None:
            raise SystemExit("--audio-dir required for a real run (see module docstring BLOCKER) -- use --build-manifest to just check the dataset schema")
        tag = args.model_tag or (f"{MODEL_TAG}-{Path(args.lora_checkpoint).name}" if args.lora_checkpoint else MODEL_TAG)
        run(args.category, args.audio_dir, args.model_name, args.out_dir, args.limit,
            lora_checkpoint=args.lora_checkpoint, model_tag=tag)

# Tonight: Azure test plan

## Goal for tonight

Do **not** attempt the full evidence-grounded assistant or claim real-music
transfer tonight. Establish two things cleanly:

1. What the base audio LLM can hear on independent, real/licensed evaluation audio.
2. Whether the new self-transcription curriculum (Track Z) trains and evaluates
   successfully on GPU.

This gives a defensible baseline plus the first execution of the intervention most
aligned with the new research direction. It does not require new recordings or
human annotation.

## First: Azure eligibility gate

Check **Azure Portal → Subscriptions → Usage + quotas → Compute** in the intended
region *before* provisioning anything. Search `NC`, `ND`, `H100`, and `H200`.

An Azure for Students subscription commonly has zero quota for the GPU N-series and
may not permit a quota increase. If that is the case, the student credit cannot run
this experiment; use a university-sponsored/paid Azure subscription with N-series
quota, or another GPU provider. Azure Machine Learning does not bypass this quota.

## Hardware choice (if GPU quota is available)

Use a Linux NVIDIA GPU VM with **one** large GPU. The present scripts load
Qwen2.5-Omni-7B in bf16 without quantization and do not distribute across GPUs.

- Best target: a single 94-GB H100 VM such as `Standard_NC40ads_H100_v5`, if the
  region/subscription offers it.
- A single 80-GB A100 is also a reasonable fallback.
- Do **not** request Azure's `ND96isr_H200_v5` merely because past runs used an
  H200: Azure's H200 SKU starts at **eight 141-GB GPUs**, while this code would use
  only one. It is extreme overkill and likely unaffordable for a pilot.

Verify the actual GPU and free memory first:

```bash
nvidia-smi
```

If that does not show a suitable GPU, stop before downloading models or data. Do not
change precision mid-experiment merely to fit a small GPU; that would create an
unregistered condition.

## One-time VM setup

On the VM, clone the project and create a separate environment. Authenticate to
Hugging Face before the first model/data download; the model and datasets may require
accepting their terms in the browser first.

```bash
git clone https://github.com/rupalirajesh/music-understanding.git /data/music-understanding
cd /data/music-understanding/experiments
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers accelerate peft datasets soundfile librosa
pip install -r requirements.txt
huggingface-cli login
```

Before any serious job, confirm imports and the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The WAV stimuli are deliberately excluded from Git. Generate them on the VM from the
committed, deterministic generator; do not copy the local manifests from another
machine or rebuild jobs.

```bash
sudo apt-get update
sudo apt-get install -y fluidsynth
bash scripts/00_download_soundfonts.sh
python scripts/01_generate_stimuli.py
python scripts/05_selftest.py
git diff --exit-code -- manifests/stimuli.parquet
```

The final command must report no manifest difference. If it does, stop and diagnose
before running a model; the existing jobs table is frozen and must remain aligned to
the generated audio.

## Run 1 — PitchBench smoke test (30–45 minutes)

**Question:** can the base model identify an isolated pitch on a fully independent,
controlled audio benchmark?

This is not the paper's headline test. It confirms the Azure/model pipeline and gives
an external perception reference.

```bash
python gpu/eval_pitchbench.py \
  --config pitchbench_a1_single_pitch_id --limit 5 \
  --out-dir results/external_benchmarks/pitchbench_smoke
```

Listen to or inspect the five responses. Check that the prompt, response, predicted
MIDI number, and ground truth are sensible. Only then run a modest preliminary set:

```bash
python gpu/eval_pitchbench.py \
  --config pitchbench_a1_single_pitch_id --limit 100 \
  --out-dir results/external_benchmarks/pitchbench_a1_100
```

Use different output directories for smoke and preliminary runs: this script is
resumable and will skip an output file that already exists.

**Success:** a results JSON appears and the accuracy is not a parsing artifact.
**Do not conclude:** anything about broad music understanding from isolated notes.

## Run 2 — CMI-Bench smoke test (real audio; start while Run 1 is checked)

**Question:** can the same base model perform on independent real music tasks—key,
pitch, beat, melody, and vocal/instrument technique?

CMI-Bench is the useful external breadth check. Its real-audio test bundle is larger,
so keep it outside this repository.

```bash
git clone https://github.com/nicolaus625/CMI-bench.git /data/CMI-bench
cd /data/CMI-bench
wget https://huggingface.co/datasets/nicolaus625/CMI-bench/resolve/main/test_Data.zip
unzip test_Data.zip -d test

cd /path/to/music-understanding/experiments
python gpu/eval_cmibench.py \
  --cmi-bench-dir /data/CMI-bench --limit 5 \
  --output-dir /data/cmi_smoke_results
```

Manually inspect responses before the full priority subset. If the audio path and
responses look right:

```bash
python gpu/eval_cmibench.py --cmi-bench-dir /data/CMI-bench
cd /data/CMI-bench
python evaluate.py --model qwen25omni --task all
```

The script's default seven-task subset is intentional: it covers key, pitch, melody,
beat, voice technique, instrument technique, and real multitrack music. It is a much
better use of the GPU than running every available task blindly.

**Success:** a scored baseline across independent real audio.
**Caveat:** it is a capability baseline, not proof of grounded explanations and not
yet a transfer experiment.

## Run 3 — Track Z smoke test (the important intervention)

**Question:** can teaching the model to produce a simple onset/duration/pitch event
list during fine-tuning improve its subsequent audio-question performance?

This is the closest existing experiment to the new central hypothesis. It trains on
the current controlled battery, using two objectives on the same audio: answer the
question (60%) or make an audio-derived event-list transcription (40%). At evaluation
it sees only the ordinary question, so it cannot simply copy the event list.

First, run exactly eight training steps:

```bash
python gpu/train_track_z_transcribe.py --seed 0 --smoke-test
```

Only proceed if the loss is finite and decreases or is at least stable, the LoRA
target modules are found, and no audio/template error appears. Then launch the full
one-seed run overnight:

```bash
python gpu/train_track_z_transcribe.py --seed 0
```

**Do not report a result from this alone.** Tomorrow, compare its held-out answer
performance with the matched base-model rows and re-probe the audio encoder. The
publication-quality result requires three seeds and real-audio transfer.

## Optional Run 4 — narrow real-timbre transfer check

Run this only if the saved Track-D-zoom adapter is actually available on the VM.
It tests whether a pitch-trace-plus-reference interface transfers from synthetic tones
to real NSynth instrument recordings. It does **not** test polyphonic songs, harmony,
or a general visual-DSP claim.

```bash
python -m musicprobe.real_music_nsynth --n 60 --seed 0
python -m musicprobe.real_music_nsynth --dzoom-jobs --seed 0
python gpu/eval_track_dzoom_real.py --seed 0 --limit 40 --no-lora
python gpu/eval_track_dzoom_real.py --seed 0 --limit 40
```

Run both halves against the same jobs. If the checkpoint is absent, skip this run;
do not retrain it tonight.

## Explicitly do not run tonight

- Tracks X/Y or another generic image/DSP representation sweep. Existing results
  already make this low-priority under the new framing.
- BASS: its audio files are unresolved.
- A full all-30-config PitchBench run: several schemas remain unverified; begin with
  the audited A1 configuration.
- Any claim about all genres/cultures, broad music understanding, or user-facing
  teaching quality.

## What to save and send back

For each completed run, save:

1. exact command and model revision;
2. GPU type and peak memory from `nvidia-smi`;
3. raw result file(s), not just an accuracy screenshot;
4. five manually checked examples (audio/task/answer/ground truth);
5. errors, warnings, and runtime.

That is enough to decide the next experiment without mistaking a plumbing problem
for a model result.

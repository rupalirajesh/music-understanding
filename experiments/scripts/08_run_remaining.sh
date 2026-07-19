#!/usr/bin/env bash
# One-shot H100 runbook: everything that hasn't run yet, in dependency order.
#
#   bash scripts/08_run_remaining.sh
#
# Every step is resumable/idempotent (runners skip finished job_ids, extraction
# skips existing .npz, scoring is a pure function of responses) — if anything
# crashes, just rerun this script; it continues where it stopped.
# Full output is logged to results/runlogs/ (tracked in git).
#
# Prereqs on the box (one-time):
#   pip install -r requirements.txt torch transformers accelerate soundfile pandas pyarrow librosa openpyxl scikit-learn torchaudio
#   bash scripts/00_download_soundfonts.sh
#   python scripts/01_generate_stimuli.py          # regenerate WAVs (~12 min) if not present
#   # AF3/Music Flamingo only:
#   #   git clone https://github.com/NVIDIA/audio-flamingo -b audio_flamingo_3 && pip install -e audio-flamingo
#   # Gemini top-up only: export PORTKEY_API_KEY=...
#   # GPT-4o-audio only:  export OPENAI_API_KEY=...

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

# ---------- Track A: behavioral battery ------------------------------------
# Already-run models first: they only pick up the new instrument_id jobs.
step "Track A / local: Qwen2-Audio instrument_id top-up"
try $PY -m musicprobe.runners.run_local --model Qwen/Qwen2-Audio-7B-Instruct

step "Track A / local: Qwen2.5-Omni (smoke-test 5, then full)"
try $PY -m musicprobe.runners.run_local --model Qwen/Qwen2.5-Omni-7B --limit 5
try $PY -m musicprobe.runners.run_local --model Qwen/Qwen2.5-Omni-7B

step "Track A / local: Audio Flamingo 3 (needs NVIDIA llava fork, see prereqs)"
try $PY -m musicprobe.runners.run_local --model nvidia/audio-flamingo-3 --limit 5
try $PY -m musicprobe.runners.run_local --model nvidia/audio-flamingo-3

# Uncomment when ready (verify HF repo id for Music Flamingo; Qwen3-Omni needs
# transformers>=4.57 and ~60GB):
# try $PY -m musicprobe.runners.run_local --model nvidia/music-flamingo
# try $PY -m musicprobe.runners.run_local --model Qwen/Qwen3-Omni-30B-A3B-Instruct

step "Track A / API: Gemini instrument_id top-up"
if [ -n "${PORTKEY_API_KEY:-}" ]; then
  try $PY -m musicprobe.runners.run_api --model portkey-gemini-2.5-pro
else echo "PORTKEY_API_KEY not set — skipping Gemini top-up"; fi

step "Track A / API: GPT-4o-audio full battery"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  try $PY -m musicprobe.runners.run_api --model gpt-4o-audio-preview
else echo "OPENAI_API_KEY not set — skipping GPT-4o-audio"; fi

step "Track A: score + review-export every model that has responses"
for f in results/trackA/responses__*.parquet; do
  m=$(basename "$f" .parquet); m=${m#responses__}
  case "$m" in dry|selftest*) continue;; esac
  try $PY scripts/04_export_for_review.py --model "$m"
done
try $PY scripts/06_export_excel.py

# ---------- Track B: representations + routing -----------------------------
step "Track B: encoder activations (music-SSL vs ASR vs contrastive)"
try $PY gpu/extract_activations.py --model m-a-p/MERT-v1-330M      --out acts/mert330
try $PY gpu/extract_activations.py --model openai/whisper-large-v3 --out acts/whisper
try $PY gpu/extract_activations.py --model laion/clap-htsat-unfused --out acts/clap

step "Track B: linear probe suite (results/trackB/probes/)"
PROBES="pitch_note_id:pitch_class octave_id:octave instrument_id:program \
interval_id:ground_truth key_id:ground_truth mode_id:ground_truth \
chord_quality:quality note_count:ground_truth beats_per_bar:ground_truth \
tuning_judgment:ground_truth cents_discrimination:ground_truth"
for enc in mert330 whisper clap; do
  for p in $PROBES; do
    try $PY gpu/probe.py --acts "acts/$enc" --task "${p%%:*}" --target "${p##*:}"
  done
done

step "Track B: audio-token attention diagnostic (results/trackB/attention/)"
try $PY gpu/attention_audio.py --model Qwen/Qwen2-Audio-7B-Instruct

# ---------- Generation battery ---------------------------------------------
step "Genmodel: MusicGen constraint-adherence battery"
try $PY genmodel/run_musicgen.py --model facebook/musicgen-medium
try $PY genmodel/score_gen.py --dir genmodel/outputs

# ---------- Ship it ---------------------------------------------------------
step "Commit + push results"
git add results/
git commit -m "Full battery: remaining models + instrument_id + Track B + genmodel" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines"

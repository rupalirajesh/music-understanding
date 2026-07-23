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
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow librosa openpyxl scikit-learn torchaudio openai portkey_ai
#   bash scripts/00_download_soundfonts.sh
#   python scripts/01_generate_stimuli.py          # regenerate WAVs (~12 min) if not present
#   export PORTKEY_API_KEY=...   # REQUIRED (Gemini instrument_id top-up)
#   export OPENAI_API_KEY=...    # REQUIRED (GPT-4o-audio full battery)
# All five local models run on plain transformers (>=5.14, auto-installed
# below) with checkpoint ids verified on the HF hub — no vendor forks.

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

# ---------- Preflight: refuse to start half-blind --------------------------
step "Preflight: API keys (hard requirement — nothing gets silently skipped)"
missing=0
[ -z "${PORTKEY_API_KEY:-}" ] \
  && { echo "PORTKEY_API_KEY not set — needed for the Gemini instrument_id top-up"; missing=1; }
[ -z "${OPENAI_API_KEY:-}" ] \
  && { echo "OPENAI_API_KEY not set — needed for the GPT-4o-audio battery"; missing=1; }
if [ "$missing" -eq 1 ]; then
  echo "STOPPING: export the key(s) above, then rerun this script."
  exit 1
fi

step "Preflight: transformers new enough for all five model families"
$PY -m pip install -q -U "transformers>=5.14" \
  || echo "!! pip upgrade failed — check manually that transformers>=5.14"

# ---------- Track A: behavioral battery ------------------------------------
# Already-run models first: they only pick up the new instrument_id jobs.
step "Track A / local: Qwen2-Audio instrument_id top-up"
try $PY -m musicprobe.runners.run_local --model Qwen/Qwen2-Audio-7B-Instruct

# Each new model: --limit 5 smoke run first — EYEBALL those 5 responses in
# results/trackA before the full pass burns hours.
for M in Qwen/Qwen2.5-Omni-7B nvidia/audio-flamingo-3-hf \
         nvidia/music-flamingo-2601-hf Qwen/Qwen3-Omni-30B-A3B-Instruct; do
  step "Track A / local: $M (smoke-test 5, then full)"
  try $PY -m musicprobe.runners.run_local --model "$M" --limit 5
  try $PY -m musicprobe.runners.run_local --model "$M"
done

step "Track A / API: Gemini instrument_id top-up"
try $PY -m musicprobe.runners.run_api --model portkey-gemini-2.5-pro

step "Track A / API: GPT-4o-audio full battery"
try $PY -m musicprobe.runners.run_api --model gpt-4o-audio-preview

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

step "Track B: each LALM's OWN encoder, re-probed on key_id/mode_id/chord_quality/interval_id"
# PROFESSOR_UPDATE.md open question #2: generic MERT/Whisper/CLAP above are
# NOT necessarily representative of what these models actually hear through
# (Music Flamingo/AF3 use AF-Whisper; Qwen-Omni uses its own AuT-derived
# tower) — extract_activations.py --own-encoder taps the real one.
# UNVERIFIED loaders (see gpu/extract_activations.py docstring) — smoke-check
# the printed "using submodule at '...'" line on first run before trusting it.
declare -A OWN_ENC=(
  [Qwen/Qwen2.5-Omni-7B]=qwen25omni_own
  [Qwen/Qwen3-Omni-30B-A3B-Instruct]=qwen3omni_own
  [nvidia/audio-flamingo-3-hf]=af3_own
  [nvidia/music-flamingo-2601-hf]=musicflamingo_own
)
OWN_PROBES="key_id:ground_truth mode_id:ground_truth chord_quality:quality interval_id:ground_truth"
for M in "${!OWN_ENC[@]}"; do
  enc="${OWN_ENC[$M]}"
  try $PY gpu/extract_activations.py --model "$M" --own-encoder --out "acts/$enc"
  for p in $OWN_PROBES; do
    try $PY gpu/probe.py --acts "acts/$enc" --task "${p%%:*}" --target "${p##*:}"
  done
done

step "Track B: audio-token attention diagnostic (results/trackB/attention/)"
# Qwen2-Audio already done; smoke-test each new model with --per-task 1
# before the full --per-task 6 pass (UNVERIFIED loaders, see attention_audio.py).
try $PY gpu/attention_audio.py --model Qwen/Qwen2-Audio-7B-Instruct
for M in Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct \
         nvidia/audio-flamingo-3-hf nvidia/music-flamingo-2601-hf; do
  try $PY gpu/attention_audio.py --model "$M" --per-task 1
  try $PY gpu/attention_audio.py --model "$M"
done

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

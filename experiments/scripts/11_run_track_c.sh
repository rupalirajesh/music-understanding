#!/usr/bin/env bash
# One-shot H100 runbook, scoped to exactly two things (2026-07-24):
#   1. Re-run the attention diagnostic on the 4 newer open models — the
#      2026-07-24 run is RETRACTED (see PROJECT_STATE.md Known gaps), it
#      never checked whether eager attention actually took effect.
#      gpu/attention_audio.py now hard-fails via assert_eager_attention()
#      if that happens again, instead of silently producing untrustworthy
#      numbers a second time.
#   2. Track C: 3-arm LoRA fine-tune on Audio-Flamingo-3
#      (gpu/train_track_c.py) on the alignment-fixable shortlist
#      (octave_id, tuning_judgment, cents_discrimination, note_count).
#
# Deliberately does NOT re-run Track A/B baseline work already done and
# committed — see scripts/08_run_remaining.sh for the full-battery runbook
# if that's ever needed again. This script is scoped so it doesn't burn
# GPU hours re-doing finished work.
#
#   bash scripts/11_run_track_c.sh
#
# Every step is smoke-tested first and logged to results/runlogs/ (tracked
# in git) — if anything crashes, read the log, fix, rerun; earlier steps are
# cheap to redo (attention diagnostic overwrites its own output files; LoRA
# training checkpoints are per-arm directories, safe to rerun one arm at a
# time with --arm <name>).
#
# Prereqs on the box (one-time):
#   pip install -r requirements.txt torch accelerate soundfile pandas pyarrow \
#     librosa openpyxl scikit-learn torchaudio peft
#   python scripts/01_generate_stimuli.py   # regenerate WAVs if not present

cd "$(dirname "$0")/.."
PY=${PY:-python}
mkdir -p results/runlogs
LOG="results/runlogs/run_trackc_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== $* ====="; }
try()  { "$@" || echo "!! step failed (continuing): $*"; }

step "Preflight: peft installed (Track C needs it, not in the original requirements)"
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q -U peft

# ---------- Part 1: attention-diagnostic re-run (bug fix, see docstring) ---
step "Attention diagnostic re-run: 4 models, smoke-test first (assert_eager_attention will hard-fail here if the bug recurs)"
for M in Qwen/Qwen2.5-Omni-7B Qwen/Qwen3-Omni-30B-A3B-Instruct \
         nvidia/audio-flamingo-3-hf nvidia/music-flamingo-2601-hf; do
  step "attention_audio.py smoke test: $M"
  $PY gpu/attention_audio.py --model "$M" --per-task 1
  # ^ NOT wrapped in try(): if this fails, STOP and read the assertion
  # message before burning GPU time on the full pass — it will name exactly
  # which submodule didn't resolve to eager attention.
  step "attention_audio.py full pass: $M"
  try $PY gpu/attention_audio.py --model "$M"
done

# ---------- Part 2: Track C — 3-arm LoRA fine-tune on AF3 -------------------
step "Track C: smoke test every arm first (8 steps, 8 examples, no checkpoint saved)"
for ARM in llm_only llm_encoder control; do
  step "train_track_c.py --arm $ARM --smoke-test"
  $PY gpu/train_track_c.py --arm "$ARM" --smoke-test
  # ^ also not wrapped in try(): eyeball the printed loss + trainable-param
  # count for each arm before committing to the full run below.
done

step "Track C: full training + held-out eval, all 3 arms"
for ARM in llm_only llm_encoder control; do
  step "train_track_c.py --arm $ARM (full)"
  try $PY gpu/train_track_c.py --arm "$ARM"
done

step "Track C: score each arm against its held-out split"
for ARM in llm_only llm_encoder control; do
  try $PY -m musicprobe.scoring --model "af3-lora-$ARM"
  try $PY scripts/04_export_for_review.py --model "af3-lora-$ARM"
done

# ---------- Ship it ----------------------------------------------------------
step "Commit + push results (checkpoints in gpu/track_c_checkpoints/ are gitignored — only responses/scores go up)"
git add results/ experiments/results/ 2>/dev/null
git commit -m "Track C: 3-arm LoRA on AF3 + re-verified attention diagnostic" \
  && git push || echo "nothing to commit or push failed — check manually"
step "DONE — check the log above for '!! step failed' lines, and compare af3-lora-* accuracy against results/trackA/responses__nvidia_audio-flamingo-3-hf.parquet's pre-fine-tune baseline"

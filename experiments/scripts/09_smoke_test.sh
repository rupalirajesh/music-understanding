#!/usr/bin/env bash
# Preflight smoke test — run this before scripts/08_run_remaining.sh (the
# full-battery runbook, needs PORTKEY_API_KEY). NOT required before
# scripts/11_run_track_c.sh or scripts/13_run_track_d.sh (2026-07-25) — those
# are pure local GPU work, no API keys involved; go straight to those if
# that's what you're running. Sections 1-5 below (deps, GPU, pipeline
# integrity) are still worth checking either way.
#
#   bash scripts/09_smoke_test.sh
#
# Verifies the environment and the whole pipeline end to end WITHOUT loading
# any model or spending API money (~2 min). Fix anything it flags, rerun until
# it passes, then start the real runbook.
#
# The committed manifests/jobs.parquet is the single source of truth — never
# rebuild it (jobs.build_jobs refuses while responses exist; the from-scratch
# build script was removed on purpose).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
fail=0
ok()   { echo "  [ok] $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }
warn() { echo "  [warn] $*"; }

echo "1. python deps"
for m in numpy pandas pyarrow soundfile librosa sklearn openpyxl; do
  $PY -c "import $m" 2>/dev/null && ok "$m" || bad "$m — pip install $m"
done
for m in torch transformers accelerate; do
  $PY -c "import $m" 2>/dev/null && ok "$m" || bad "$m — needed for local models + Track B"
done

echo "2. fluidsynth + soundfonts"
command -v fluidsynth >/dev/null 2>&1 && ok "fluidsynth" \
  || bad "fluidsynth — apt-get install -y fluidsynth (or conda -c conda-forge)"
$PY - <<'EOF' || fail=1
from musicprobe.config import available_soundfonts
sfs = available_soundfonts()
assert len(sfs) == 3, f"[FAIL] {len(sfs)}/3 soundfonts — bash scripts/00_download_soundfonts.sh"
print(f"  [ok] 3/3 soundfonts")
EOF

echo "3. stimuli + manifest + jobs"
$PY - <<'EOF' || fail=1
import pandas as pd
from musicprobe.config import EXP_ROOT, MANIFEST_PATH
from musicprobe.jobs import JOBS_PATH
man = pd.read_parquet(MANIFEST_PATH)
jobs = pd.read_parquet(JOBS_PATH)
missing = [p for p in man.audio_path.unique() if not (EXP_ROOT / p).exists()]
assert not missing, (f"[FAIL] {len(missing)} WAVs missing — regenerate with: "
                    "python scripts/01_generate_stimuli.py  (jobs.parquet is "
                    "committed; never rebuild it)")
assert len(jobs) == 2208, f"[FAIL] jobs.parquet has {len(jobs)} rows, expected 2208 — wrong checkout? never rebuild it"
assert jobs.task.nunique() == 13, f"[FAIL] expected 13 tasks, got {jobs.task.nunique()}"
print(f"  [ok] {len(man)} stimuli, all WAVs on disk; {len(jobs)} jobs / 13 tasks (incl. instrument_id)")
EOF

echo "4. pipeline self-test (dry backend, end to end: run -> score -> export)"
$PY scripts/05_selftest.py >/dev/null 2>&1 && ok "selftest passed" \
  || bad "selftest — run  python scripts/05_selftest.py  to see which check broke"

echo "5. GPU"
$PY -c "import torch; assert torch.cuda.is_available(); \
print('  [ok] ' + torch.cuda.get_device_name(0))" 2>/dev/null \
  || warn "no CUDA GPU visible — local models + Track B will not run here"

echo "6. API keys (only needed for scripts/08_run_remaining.sh — NOT for Track C/D)"
[ -n "${PORTKEY_API_KEY:-}" ] && ok "PORTKEY_API_KEY set (Gemini top-up)" \
  || warn "PORTKEY_API_KEY not set — fine if you're only running Track C/D" \
          "(scripts/11_run_track_c.sh, scripts/13_run_track_d.sh); export it" \
          "first if you're running scripts/08_run_remaining.sh"
# GPT-4o-audio: OUT OF SCOPE (no OpenAI API access) — removed 2026-07-25,
# no longer checked here or run by 08_run_remaining.sh.

echo
if [ "$fail" -eq 0 ]; then
  echo "SMOKE TEST PASSED. Pending work, pick what you're running:"
  echo "  bash scripts/11_run_track_c.sh   # Track C: 3-arm LoRA on AF3 (no API key needed)"
  echo "  bash scripts/13_run_track_d.sh   # Track D Phase 1: Qwen2.5-Omni-7B + spectrogram-image (no API key needed)"
  echo "  bash scripts/08_run_remaining.sh # full-battery runbook (needs PORTKEY_API_KEY) — only if that's what you mean to run"
else
  echo "SMOKE TEST FAILED — fix the [FAIL] lines above and rerun"
fi
exit $fail

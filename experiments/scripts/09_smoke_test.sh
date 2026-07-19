#!/usr/bin/env bash
# Preflight smoke test — run this BEFORE scripts/08_run_remaining.sh.
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

echo "6. API keys (REQUIRED — the runbook refuses to start without them)"
[ -n "${PORTKEY_API_KEY:-}" ] && ok "PORTKEY_API_KEY set (Gemini top-up)" \
  || bad "PORTKEY_API_KEY not set — export it before running 08"
[ -n "${OPENAI_API_KEY:-}" ] && ok "OPENAI_API_KEY set (GPT-4o-audio)" \
  || bad "OPENAI_API_KEY not set — export it before running 08"

echo
if [ "$fail" -eq 0 ]; then
  echo "SMOKE TEST PASSED — now run:  bash scripts/08_run_remaining.sh"
else
  echo "SMOKE TEST FAILED — fix the [FAIL] lines above and rerun"
fi
exit $fail

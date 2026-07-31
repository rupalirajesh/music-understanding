"""Render the reftone/wrong_reftone WAV variants for every tuning_judgment
stimulus (see musicprobe/reftone.py). Pure numpy synthesis (reuses
synth.tone_pair, same building block as cents_discrimination) — cheap,
deterministic, CPU-only.

  python scripts/render_reftones.py
  python scripts/render_reftones.py --limit 20 --force
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from musicprobe.config import EXP_ROOT, MANIFEST_PATH
from musicprobe.reftone import (build_reftone_audio, build_wrong_reftone_audio,
                                reftone_path, wrong_reftone_path)
from musicprobe.synth import write_wav


def main(manifest, exp_root, force, limit):
    root = Path(exp_root)
    man = pd.read_parquet(manifest)
    man = man[man.task == "tuning_judgment"]
    if limit:
        man = man.head(limit)
    done = skipped = failed = 0
    for i, row in enumerate(man.itertuples(), 1):
        factors = json.loads(row.factors) if isinstance(row.factors, str) else row.factors
        base_midi, midi_exact = factors["base_midi"], factors["midi_exact"]
        ref_out = root / reftone_path(row.audio_path)
        wrong_out = root / wrong_reftone_path(row.audio_path)
        if ref_out.exists() and wrong_out.exists() and not force:
            skipped += 1
            continue
        try:
            if force or not ref_out.exists():
                write_wav(ref_out, build_reftone_audio(base_midi, midi_exact))
            if force or not wrong_out.exists():
                seed = zlib.crc32(f"wrong_reftone|{row.stimulus_id}".encode())
                write_wav(wrong_out, build_wrong_reftone_audio(base_midi, midi_exact, seed))
            done += 1
        except Exception as e:
            failed += 1
            print(f"!! failed on {row.stimulus_id}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            print(f"{i}/{len(man)} (rendered {done}, skipped {skipped}, failed {failed})")
    print(f"done: rendered {done}, skipped {skipped}, failed {failed}, out of {len(man)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.manifest, args.exp_root, args.force, args.limit)

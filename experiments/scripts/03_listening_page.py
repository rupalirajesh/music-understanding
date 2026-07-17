"""Build listening.html — a local page for auditing stimuli by ear.

  .venv/bin/python scripts/03_listening_page.py --per-task 12
  open listening.html

Audio players grouped by task, each with its ground truth hidden behind a
click (so you can test yourself blind first) plus all factors. Samples are
randomly drawn but seeded — rerun with --seed to get a fresh draw.
"""
import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from musicprobe.config import EXP_ROOT
from musicprobe.manifest import load_manifest


def main(per_task: int, seed: int):
    man = load_manifest()
    rng = np.random.default_rng(seed)
    parts = ["""<meta charset='utf-8'><title>Stimulus audit</title>
<style>
 body{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}
 h2{border-bottom:2px solid #ccc;padding-bottom:.2rem;margin-top:2.5rem}
 .item{display:flex;align-items:center;gap:1rem;padding:.4rem 0;border-bottom:1px solid #eee}
 .meta{font-size:.85rem;color:#555;flex:1}
 details{min-width:14rem} summary{cursor:pointer;color:#0366d6}
 audio{width:280px}
</style>
<h1>Stimulus audit page</h1>
<p>Click "reveal answer" only after listening — audit both the audio quality
and whether YOU agree with the ground truth. Flag anything ambiguous.</p>"""]
    for task, g in man.groupby("task"):
        take = g.sample(min(per_task, len(g)), random_state=int(rng.integers(2**31)))
        parts.append(f"<h2>{task} <small>({len(g)} total, showing {len(take)})</small></h2>")
        for r in take.itertuples():
            facts = ", ".join(f"{k}={v}" for k, v in r.factors.items())
            parts.append(
                f"<div class='item'><audio controls preload='none' "
                f"src='{html.escape(r.audio_path)}'></audio>"
                f"<details><summary>reveal answer</summary>"
                f"<b>{html.escape(str(r.ground_truth))}</b></details>"
                f"<div class='meta'>{html.escape(r.stimulus_id)}<br>{html.escape(facts)}</div></div>")
    out = EXP_ROOT / "listening.html"
    out.write_text("\n".join(parts))
    print(f"wrote {out} — open it in a browser (audio paths are relative,"
          " so keep it inside experiments/)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args.per_task, args.seed)

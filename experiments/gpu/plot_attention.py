"""Plot the audio-attention diagnostic: how much of the LM's attention lands on
the audio tokens, by generation step and by layer, per model.

Reads results/trackB/attention/attn__<model>.parquet (long: job_id, task, step,
layer, attn_audio_frac, uniform_frac) and writes a two-panel figure:
  left  = mean audio-attention vs generation step (the "decay" curve)
  right = mean audio-attention vs layer
The dashed line is the uniform-attention baseline (n_audio_tokens / context_len):
above it = the model looks at audio more than chance; on it = audio is furniture.

  python gpu/plot_attention.py
"""
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musicprobe.config import TRACKB_DIR

ATTN_DIR = Path(TRACKB_DIR) / "attention"
COLORS = ["#2a78d6", "#e8710a", "#0a9648", "#c0392b", "#7d3ac1", "#00838f"]


def short(name):
    return name.replace("responses__", "").replace("attn__", "")


def main():
    files = sorted(glob.glob(str(ATTN_DIR / "attn__*.parquet")))
    if not files:
        print("no attn__*.parquet found"); return
    fig, (axs, axl) = plt.subplots(1, 2, figsize=(13, 5))
    uni_step, uni_layer = [], []
    for i, f in enumerate(files):
        m = short(os.path.basename(f)[:-len(".parquet")])
        d = pd.read_parquet(f)
        c = COLORS[i % len(COLORS)]
        by_step = d.groupby("step")[["attn_audio_frac", "uniform_frac"]].mean()
        by_layer = d.groupby("layer")[["attn_audio_frac", "uniform_frac"]].mean()
        axs.plot(by_step.index, by_step.attn_audio_frac, "-o", ms=3, color=c, label=m)
        axl.plot(by_layer.index, by_layer.attn_audio_frac, "-", lw=1.6, color=c, label=m)
        uni_step.append(by_step.uniform_frac)
        uni_layer.append(by_layer.uniform_frac)
    # single uniform baseline (averaged across models)
    us = pd.concat(uni_step, axis=1).mean(axis=1)
    ul = pd.concat(uni_layer, axis=1).mean(axis=1)
    axs.plot(us.index, us.values, "k--", lw=1.2, alpha=.6, label="uniform baseline")
    axl.plot(ul.index, ul.values, "k--", lw=1.2, alpha=.6, label="uniform baseline")

    axs.set_xlabel("generation step"); axs.set_ylabel("mean attention on audio tokens")
    axs.set_title("Audio-attention vs generation step\n(decay = listens first, then coasts)")
    axl.set_xlabel("layer"); axl.set_ylabel("mean attention on audio tokens")
    axl.set_title("Audio-attention vs layer")
    for ax in (axs, axl):
        ax.grid(alpha=.25); ax.set_ylim(bottom=0)
    axs.legend(fontsize=8)
    fig.suptitle("How much do audio LLMs attend to the audio tokens? "
                 "(eager attention, per-task 6)", fontsize=12)
    fig.tight_layout()
    out = ATTN_DIR / "attention_graph.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    # also print the numbers
    print("\nmean audio-attention (all steps/layers) vs uniform baseline:")
    for f in files:
        d = pd.read_parquet(f)
        print(f"  {short(os.path.basename(f)[:-len('.parquet')]):40} "
              f"audio={d.attn_audio_frac.mean():.4f}  uniform={d.uniform_frac.mean():.4f}  "
              f"ratio={d.attn_audio_frac.mean()/d.uniform_frac.mean():.2f}x")


if __name__ == "__main__":
    main()

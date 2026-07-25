"""
Rebuild Track B analysis CSVs + plots, extended with the attention diagnostic
and own-encoder probes Sethu ran for the 4 remaining open models (Qwen2.5-Omni,
Qwen3-Omni, Audio-Flamingo-3, Music-Flamingo).

Outputs (experiments/results/trackB/analysis/):
  trackB_attention_all.csv        - attn-on-audio-tokens by layer, all 5 models
  trackB_attention_by_model.png   - 10-task grid, one line per model
  trackB_probes_all.csv           - probe accuracy by layer, all encoders incl. "own"
  trackB_probes_own_encoder.png   - 4-task grid, own-encoder probes per model
                                     (external CLAP/MERT/Whisper shown as light refs)
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "trackB"
ATTN_DIR = ROOT / "attention"
PROBE_DIR = ROOT / "probes"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "Qwen_Qwen2-Audio-7B-Instruct": "Qwen2-Audio-7B",
    "Qwen_Qwen2.5-Omni-7B": "Qwen2.5-Omni-7B",
    "Qwen_Qwen3-Omni-30B-A3B-Instruct": "Qwen3-Omni-30B-A3B",
    "nvidia_audio-flamingo-3-hf": "Audio-Flamingo-3",
    "nvidia_music-flamingo-2601-hf": "Music-Flamingo",
}
MODEL_ORDER = list(MODEL_LABELS)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODEL_COLOR = dict(zip(MODEL_ORDER, COLORS))

OWN_LABELS = {
    "qwen25omni_own": "Qwen2.5-Omni-7B (own enc.)",
    "qwen3omni_own": "Qwen3-Omni-30B-A3B (own enc.)",
    "af3_own": "Audio-Flamingo-3 (own enc.)",
    "musicflamingo_own": "Music-Flamingo (own enc.)",
}
OWN_ORDER = list(OWN_LABELS)
OWN_COLOR = dict(zip(OWN_ORDER, COLORS))

EXTERNAL_LABELS = {
    "whisper": "Whisper-enc (ASR)",
    "mert330": "MERT-330M (music-SSL)",
    "clap": "CLAP (contrastive)",
}

# ---------------------------------------------------------------- attention
attn_frames = []
for f in sorted(ATTN_DIR.glob("attn_summary__*.csv")):
    model = f.stem.replace("attn_summary__", "")
    df = pd.read_csv(f)
    df["model"] = model
    attn_frames.append(df)
attn_all = pd.concat(attn_frames, ignore_index=True)
attn_all.to_csv(OUT_DIR / "trackB_attention_all.csv", index=False)

by_layer = attn_all[attn_all["view"] == "by_layer"].copy()
by_layer["layer"] = by_layer["layer"].astype(float)
tasks = sorted(by_layer["task"].unique())

ncols = 4
nrows = -(-len(tasks) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
for i, task in enumerate(tasks):
    ax = axes[i // ncols][i % ncols]
    sub = by_layer[by_layer["task"] == task]
    baseline_done = False
    for model in MODEL_ORDER:
        ms = sub[sub["model"] == model].sort_values("layer")
        if ms.empty:
            continue
        ax.plot(ms["layer"], ms["attn_audio_frac"], color=MODEL_COLOR[model],
                label=MODEL_LABELS[model], linewidth=1.6)
        if not baseline_done:
            ax.axhline(ms["uniform_frac"].iloc[0], color="gray", linestyle="--",
                        linewidth=1, alpha=0.7)
            baseline_done = True
    ax.set_title(task, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.9)
    ax.grid(True, alpha=0.3)
for j in range(len(tasks), nrows * ncols):
    axes[j // ncols][j % ncols].axis("off")
for r in range(nrows):
    axes[r][0].set_ylabel("attn fraction on audio")
for c in range(ncols):
    axes[nrows - 1][c].set_xlabel("decoder layer")

handles = [plt.Line2D([0], [0], color=MODEL_COLOR[m], linewidth=2, label=MODEL_LABELS[m])
           for m in MODEL_ORDER]
handles.append(plt.Line2D([0], [0], color="gray", linestyle="--", label="uniform baseline"))
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.suptitle("Track B — fraction of decoder attention mass on audio tokens, by layer\n"
             "(all 5 open-weight LALMs with this diagnostic run)", fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(OUT_DIR / "trackB_attention_by_model.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------------- probes
probe_frames = []
for f in sorted(PROBE_DIR.glob("probe__*.csv")):
    m = re.match(r"probe__(.+?)__(.+?)__(.+)\.csv", f.name)
    encoder, task, target = m.groups()
    df = pd.read_csv(f)
    df["encoder"] = encoder
    df["task"] = task
    df["target"] = target
    probe_frames.append(df)
probe_all = pd.concat(probe_frames, ignore_index=True)
probe_all.to_csv(OUT_DIR / "trackB_probes_all.csv", index=False)

own_tasks = sorted({t for e, t in probe_all[["encoder", "task"]].drop_duplicates().itertuples(index=False)
                     if e in OWN_ORDER})

fig, axes = plt.subplots(1, len(own_tasks), figsize=(5 * len(own_tasks), 4.2), squeeze=False)
axes = axes[0]
for i, task in enumerate(own_tasks):
    ax = axes[i]
    sub = probe_all[probe_all["task"] == task]
    for enc in EXTERNAL_LABELS:
        es = sub[sub["encoder"] == enc].sort_values("layer")
        if es.empty:
            continue
        ax.plot(es["layer"], es["probe_acc"], color="lightgray", linewidth=1.2,
                zorder=1, label=EXTERNAL_LABELS[enc] if i == 0 else None)
    baseline_done = False
    for own in OWN_ORDER:
        os_ = sub[sub["encoder"] == own].sort_values("layer")
        if os_.empty:
            continue
        ax.plot(os_["layer"], os_["probe_acc"], color=OWN_COLOR[own], linewidth=2,
                zorder=2, label=OWN_LABELS[own] if i == 0 else None)
        if not baseline_done:
            ax.axhline(os_["chance"].iloc[0], color="black", linestyle="--",
                        linewidth=1, alpha=0.6, label="chance" if i == 0 else None)
            baseline_done = True
    ax.set_title(task, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("layer")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("probe accuracy")

fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=9)
fig.suptitle("Track B — layer-wise linear probe accuracy, own audio encoder per LALM\n"
             "(light gray = external encoders CLAP/MERT/Whisper on the same task, for reference)",
             fontsize=12)
fig.tight_layout(rect=[0, 0.08, 1, 0.93])
fig.savefig(OUT_DIR / "trackB_probes_own_encoder.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("wrote:")
for p in sorted(OUT_DIR.iterdir()):
    print(" ", p.name)

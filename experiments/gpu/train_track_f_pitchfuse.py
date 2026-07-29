"""Track F — pitch-STREAM FUSION (the scalable, end-to-end fix).

Instead of a rendered image or a summary string, inject a task-agnostic
FRAME-LEVEL pitch contour straight into the LM's embedding space via a small
learned projector, fused alongside the normal audio tokens:

  prompt gets K copies of an unused special token (<|quad_start|>) as pitch
  placeholders; a forward hook on embed_tokens replaces those K embedding rows
  with pitch_projector(pitch_features). Audio still merges normally (verified),
  and generation flows normally. Trainable: LoRA(thinker) + the projector.

Modality-dropout training (both / pitch-only / audio-only) forces the model to
sometimes rely on the pitch channel; same within-model paired eval as Track D/E
(3 seeds, McNemar, wrong-* controls) so it's directly comparable.

  python gpu/train_track_f_pitchfuse.py --seed 0 --smoke-test
  python gpu/train_track_f_pitchfuse.py --seed 0
Analyze with: gpu/analyze_track_d.py --tag pitchfuse
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied  # noqa: E402
from train_track_d_force import _split  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.pitch_feats import load_map, K as PITCH_K  # noqa: E402

PITCH_ID = 151650          # <|quad_start|> — unused in our prompts, no modality merge
PITCH_TOK = "<|quad_start|>"
HIDDEN = 3584
MODE_P = {"both": 0.5, "pitch_only": 0.25, "audio_only": 0.25}
PMAP = None; SID2AUDIO = None; PKEYS = None


def _init():
    global PMAP, SID2AUDIO, PKEYS
    if PMAP is None:
        PMAP = load_map()
        man = pd.read_parquet(MANIFEST_PATH)
        SID2AUDIO = dict(zip(man.stimulus_id, man.audio_path))
        PKEYS = sorted(PMAP)


class PitchProjector(nn.Module):
    def __init__(self, k=PITCH_K, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 256), nn.GELU(), nn.Linear(256, hidden))

    def forward(self, pf):                      # pf: (K,2) or (1,K,2) -> (K,hidden)
        return self.net(pf.reshape(-1, 2))


def _install(thinker):
    """Attach projector + embed hook. Returns nothing; thinker gains
    .pitch_projector and ._cur_pitch (set per forward)."""
    dev = next(thinker.parameters()).device
    dtype = next(thinker.parameters()).dtype
    thinker.pitch_projector = PitchProjector().to(dev, dtype)
    thinker._cur_pitch = None

    def hook(mod, inp, out):
        pf = getattr(thinker, "_cur_pitch", None)
        if pf is None:
            return out
        ids = inp[0]
        mask = (ids == PITCH_ID)
        if mask.sum() == 0:
            return out
        emb = thinker.pitch_projector(pf.to(out.dtype))       # (K,hidden)
        out = out.clone()
        out[mask] = emb[:int(mask.sum().item())]
        return out

    thinker.get_input_embeddings().register_forward_hook(hook)


class FusedWrapper(nn.Module):
    """Trainer target: stashes pitch on the thinker, then forwards normally."""
    def __init__(self, thinker):
        super().__init__(); self.thinker = thinker

    def forward(self, pitch_features=None, **kw):
        self.thinker._cur_pitch = pitch_features
        return self.thinker(**kw)


def _build_example(processor, exp_root, prompt, audio_path, pitch_feat, answer, use_audio):
    content = []
    if use_audio and isinstance(audio_path, str):
        content.append({"type": "audio", "audio": str(exp_root / audio_path)})
    text = prompt + (("\n" + PITCH_TOK * PITCH_K) if pitch_feat is not None else "")
    content.append({"type": "text", "text": text})
    pin = processor.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt")
    plen = pin["input_ids"].shape[1]
    ans = processor.tokenizer(answer + processor.tokenizer.eos_token,
                              return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([pin["input_ids"], ans], dim=1)
    labels = input_ids.clone(); labels[:, :plen] = -100
    out = dict(pin); out["input_ids"] = input_ids; out["labels"] = labels
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat([out["attention_mask"], torch.ones_like(ans)], dim=1)
    if pitch_feat is not None:
        out["pitch_features"] = torch.tensor(pitch_feat, dtype=torch.float32)
    return out


class DropoutDataset:
    def __init__(self, rows, processor, exp_root, seed):
        self.rows, self.processor, self.exp_root = rows, processor, exp_root
        self.rng = np.random.default_rng(seed)
        self.modes = list(MODE_P); self.probs = [MODE_P[m] for m in self.modes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        mode = self.modes[int(self.rng.choice(len(self.modes), p=self.probs))]
        use_audio = mode in ("both", "audio_only")
        pf = PMAP.get(r.audio_path) if mode in ("both", "pitch_only") else None
        return _build_example(self.processor, self.exp_root, r.prompt, r.audio_path,
                              pf, r.ground_truth, use_audio)


def tag(seed):
    return f"qwen25omni-pitchfuse-s{seed}"


def _make(seed):
    _init()
    model, processor, lm_path = load_qwen_omni_for_training()
    from peft import get_peft_model
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    _install(model.thinker)                     # projector requires_grad stays True
    assert_lora_applied(model.thinker, f"pitchfuse-s{seed}")
    return model, processor


def train(seed, smoke, exp_root):
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    model, processor = _make(seed)
    train_rows, held = _split(exp_root)
    n_proj = sum(p.numel() for p in model.thinker.pitch_projector.parameters())
    print(f"[pitchfuse] seed={seed}: {len(train_rows)} rows / {len(held)} held-out; "
          f"projector params={n_proj}")
    model.thinker.train()
    ds = DropoutDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_f_ckpt" / tag(seed)
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        num_train_epochs=1 if smoke else 3, max_steps=8 if smoke else -1,
        learning_rate=2e-4, bf16=True, seed=seed,
        logging_steps=1 if smoke else 10, save_strategy="no", report_to=[],
        remove_unused_columns=False, label_names=["labels"])
    Trainer(model=FusedWrapper(model.thinker), args=args, train_dataset=ds,
            data_collator=lambda b: b[0]).train()
    if smoke:
        print("[pitchfuse] smoke done"); return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    torch.save(model.thinker.pitch_projector.state_dict(), out_dir / "pitch_projector.pt")
    print(f"[pitchfuse] saved adapter + projector to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    _init()
    base, processor, lm_path = load_qwen_omni_for_training()
    ck = GPU_DIR / "track_f_ckpt" / tag(seed)
    base.thinker = PeftModel.from_pretrained(base.thinker, str(ck))
    _install(base.thinker)
    base.thinker.pitch_projector.load_state_dict(
        torch.load(ck / "pitch_projector.pt", map_location=base.device))
    _, held = _split(EXP_ROOT)
    return base, processor, held


def _cfg_for_row(row):
    """(use_audio_path, pitch_feat) for this held-out row's condition."""
    correct = SID2AUDIO.get(row.stimulus_id, row.audio_path)
    cond = row.image_condition
    if cond == "no_image":
        return row.audio_path, None
    if cond == "wrong_image":
        j = int(np.random.default_rng(abs(hash(row.job_id)) % (2**32)).integers(len(PKEYS)))
        k = PKEYS[j] if PKEYS[j] != correct else PKEYS[(j + 1) % len(PKEYS)]
        return row.audio_path, PMAP[k]
    if cond == "image_wrong_audio":
        return row.audio_path, PMAP.get(correct)      # row.audio_path is the wrong clip
    return row.audio_path, PMAP.get(correct)


def evaluate(seed, model, processor, held, exp_root):
    model.eval()
    results = []
    for row in held.itertuples():
        audio_path, pf = _cfg_for_row(row)
        content = []
        if isinstance(audio_path, str):
            content.append({"type": "audio", "audio": str(exp_root / audio_path)})
        text = row.prompt + (("\n" + PITCH_TOK * PITCH_K) if pf is not None else "")
        content.append({"type": "text", "text": text})
        inp = processor.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(model.device)
        model.thinker._cur_pitch = (torch.tensor(pf, dtype=torch.float32).to(model.device)
                                    if pf is not None else None)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=64, do_sample=False, return_audio=False)
        if isinstance(out, tuple):
            out = out[0]
        out = out[:, inp["input_ids"].shape[1]:]
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"job_id": row.job_id, "stimulus_id": row.stimulus_id,
                        "task": row.task, "image_condition": row.image_condition,
                        "model": tag(seed), "seed": seed, "raw_response": raw,
                        "error": None, "ts": time.time()})
    out_path = RESULTS_DIR / f"responses__{tag(seed)}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[pitchfuse] seed={seed}: {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    a = ap.parse_args()
    exp_root = Path(a.exp_root)
    if a.eval_only:
        m, p, h = load_for_eval(a.seed); evaluate(a.seed, m, p, h, exp_root)
    else:
        m, p, h = train(a.seed, a.smoke_test, exp_root)
        if m is not None:
            evaluate(a.seed, m, p, h, exp_root)

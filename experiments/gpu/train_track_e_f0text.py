"""Track E — the SCALABLE audio-only fix: feed a pitch tracker's output as TEXT
(no image, no vision tower) and see if the model can then answer microtone
questions. This is the deployable version of the "give it the pitch it can't
hear" idea: one forward pass, works on any audio pyin handles.

Mirrors the conclusive Track-D design exactly so results are directly
comparable, with the image replaced by an F0-readout string in the prompt:
  with_f0   (image)             correct audio + pitch-tracker text
  no_f0     (no_image)          correct audio only            (paired baseline)
  wrong_f0  (wrong_image)       correct audio + a DIFFERENT stimulus's readout
  f0_wrong_audio (image_wrong_audio)  wrong audio + correct readout (substitute?)
Modality-dropout training (both / f0-only / audio-only) so the model is forced
to sometimes rely on the numbers and neither condition is out-of-distribution.

  python gpu/train_track_e_f0text.py --seed 0 --smoke-test
  python gpu/train_track_e_f0text.py --seed 0
Analyze with: gpu/analyze_track_d.py --tag f0text
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied  # noqa: E402
from train_track_d_force import _split  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.f0_text import load_map  # noqa: E402

MODE_P = {"both": 0.5, "f0_only": 0.25, "audio_only": 0.25}
F0MAP = None          # audio_path -> readout string
SID2AUDIO = None      # stimulus_id -> correct audio_path
F0KEYS = None         # sorted list for wrong-f0 draws


def _init_maps():
    global F0MAP, SID2AUDIO, F0KEYS
    if F0MAP is None:
        F0MAP = load_map()
        man = pd.read_parquet(MANIFEST_PATH)
        SID2AUDIO = dict(zip(man.stimulus_id, man.audio_path))
        F0KEYS = sorted(F0MAP)


def _prompt_with(prompt, f0_text):
    return prompt + ("\n\n" + f0_text if f0_text else "")


def _build_example(processor, exp_root, prompt, audio_path, f0_text, answer, use_audio):
    import torch
    content = []
    if use_audio and isinstance(audio_path, str):
        content.append({"type": "audio", "audio": str(exp_root / audio_path)})
    content.append({"type": "text", "text": _prompt_with(prompt, f0_text)})
    conv = [{"role": "user", "content": content}]
    pin = processor.apply_chat_template(conv, add_generation_prompt=True,
                                        tokenize=True, return_dict=True, return_tensors="pt")
    plen = pin["input_ids"].shape[1]
    ans = processor.tokenizer(answer + processor.tokenizer.eos_token,
                              return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([pin["input_ids"], ans], dim=1)
    labels = input_ids.clone(); labels[:, :plen] = -100
    out = dict(pin); out["input_ids"] = input_ids; out["labels"] = labels
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat([out["attention_mask"], torch.ones_like(ans)], dim=1)
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
        f0 = F0MAP.get(r.audio_path) if mode in ("both", "f0_only") else None
        return _build_example(self.processor, self.exp_root, r.prompt, r.audio_path,
                              f0, r.ground_truth, use_audio)


def tag(seed):
    return f"qwen25omni-f0text-s{seed}"


def train(seed, smoke, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    _init_maps()
    train_rows, held = _split(exp_root)
    print(f"[f0text] seed={seed}: {len(train_rows)} rows (dropout {MODE_P}) / {len(held)} held-out")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"f0text-s{seed}")
    model.thinker.train()
    ds = DropoutDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_e_ckpt" / tag(seed)
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke else 8,
        num_train_epochs=1 if smoke else 3, max_steps=8 if smoke else -1,
        learning_rate=2e-4, bf16=True, seed=seed,
        logging_steps=1 if smoke else 10, save_strategy="no", report_to=[],
        remove_unused_columns=False)
    Trainer(model=model.thinker, args=args, train_dataset=ds,
            data_collator=lambda b: b[0]).train()
    if smoke:
        print("[f0text] smoke done"); return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[f0text] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    _init_maps()
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(base.thinker, str(GPU_DIR / "track_e_ckpt" / tag(seed)))
    _, held = _split(EXP_ROOT)
    return base, processor, held


def _f0_for_row(row):
    """Return (f0_text, audio_path) for this held-out row's image_condition."""
    correct_audio = SID2AUDIO.get(row.stimulus_id, row.audio_path)
    cond = row.image_condition
    if cond == "no_image":
        return None, row.audio_path
    if cond == "wrong_image":                      # wrong readout, correct audio
        j = int(np.random.default_rng(abs(hash(row.job_id)) % (2**32)).integers(len(F0KEYS)))
        k = F0KEYS[j]
        if k == correct_audio:
            k = F0KEYS[(j + 1) % len(F0KEYS)]
        return F0MAP[k], row.audio_path
    if cond == "image_wrong_audio":                # correct readout, wrong audio (row.audio_path)
        return F0MAP.get(correct_audio), row.audio_path
    return F0MAP.get(correct_audio), row.audio_path  # image: correct readout + correct audio


def evaluate(seed, model, processor, held, exp_root):
    import torch
    model.eval()
    results = []
    for row in held.itertuples():
        f0_text, audio_path = _f0_for_row(row)
        content = []
        if isinstance(audio_path, str):
            content.append({"type": "audio", "audio": str(exp_root / audio_path)})
        content.append({"type": "text", "text": _prompt_with(row.prompt, f0_text)})
        inp = processor.apply_chat_template([{"role": "user", "content": content}],
                                            add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(model.device)
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
    print(f"[f0text] seed={seed}: {len(results)} responses -> {out_path}")


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

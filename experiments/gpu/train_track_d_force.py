"""Track D — MAKE the model use the image (novel-methods run).

The conclusive null (train_track_d_conclusive.py) showed the model ignores the
image because (a) the spectrogram is barely readable to the vision tower and
(b) audio alone already solves the tasks, so nothing forces image use. This run
fixes BOTH, then re-runs the SAME conclusive paired eval so results are directly
comparable:

  READABILITY  -> use the F0-CONTOUR image (musicprobe/f0_contour.py): the
                  vision capacity probe (gpu/probe_vision_pitch.py) showed cents
                  direction goes chance->significant and octave 0.83->0.97 when
                  we swap spectrogram -> F0 chart.
  USAGE        -> MODALITY-DROPOUT training: each training example is sampled as
                  audio+image / image-only / audio-only. The image-only mode
                  makes the image NECESSARY, so the model must build an
                  image->answer pathway; audio-only keeps the audio pathway and
                  is the in-distribution eval baseline.

  python gpu/train_track_d_force.py --seed 0 --smoke-test
  python gpu/train_track_d_force.py --seed 0
  python gpu/train_track_d_force.py --seed 0 --eval-only

Eval writes responses__qwen25omni-force-s{seed}.parquet (image paths = F0
contours); analyze with gpu/analyze_track_d.py --tag force.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
from train_track_d import load_qwen_omni_for_training, build_lora_config  # noqa: E402
from train_track_c import assert_lora_applied, _held_out_mask  # noqa: E402
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.image_jobs import IMAGE_JOBS_PATH, DEFAULT_TASKS, build_image_jobs  # noqa: E402
from musicprobe.f0_contour import f0_contour_path, f0_zoom_path  # noqa: E402

# training-example modality mix (per step): both / image-only / audio-only
MODE_P = {"both": 0.5, "image_only": 0.25, "audio_only": 0.25}

# image kind is set in __main__: "f0contour" (fixed axis) or "f0zoom" (cents-scale)
IMAGE_KIND = "f0contour"
_PATH_FN = {"f0contour": f0_contour_path, "f0zoom": f0_zoom_path}
_DIRNAME = {"f0contour": "f0contours", "f0zoom": "f0zoom"}
_RUN = {"f0contour": "force", "f0zoom": "zoom"}


def _f0_for(audio_path):
    return _PATH_FN[IMAGE_KIND](audio_path)


def _build_example(processor, exp_root, prompt, audio_path, image_path, answer,
                   use_audio=True, use_image=True):
    import torch
    content = []
    if use_audio and isinstance(audio_path, str):
        content.append({"type": "audio", "audio": str(exp_root / audio_path)})
    if use_image and isinstance(image_path, str):
        content.append({"type": "image", "image": str(exp_root / image_path)})
    content.append({"type": "text", "text": prompt})
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
    """Modality-dropout over the `image` rows: audio+F0image / image-only /
    audio-only, sampled per example. Image is the F0-contour (readable)."""
    def __init__(self, rows, processor, exp_root, seed):
        import numpy as np
        self.rows, self.processor, self.exp_root = rows, processor, exp_root
        self.rng = np.random.default_rng(seed)
        self.modes = list(MODE_P); self.probs = [MODE_P[m] for m in self.modes]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        mode = self.modes[int(self.rng.choice(len(self.modes), p=self.probs))]
        use_audio = mode in ("both", "audio_only")
        use_image = mode in ("both", "image_only")
        return _build_example(self.processor, self.exp_root, r.prompt, r.audio_path,
                              _f0_for(r.audio_path), r.ground_truth,
                              use_audio=use_audio, use_image=use_image)


def _split(exp_root):
    if not IMAGE_JOBS_PATH.exists():
        build_image_jobs()
    jobs = pd.read_parquet(IMAGE_JOBS_PATH)
    man = pd.read_parquet(MANIFEST_PATH)[["stimulus_id", "factors"]]
    wf = jobs.merge(man, on="stimulus_id", how="left")
    ho = _held_out_mask(wf).values
    # train from the `image` rows (dropout dataset re-derives modality per step)
    train = jobs[(~ho) & (jobs.image_condition == "image")]
    held = jobs[ho]  # all 4 conditions, eval
    return train.reset_index(drop=True), held.reset_index(drop=True)


def tag(seed):
    return f"qwen25omni-{_RUN[IMAGE_KIND]}-s{seed}"


def train(seed, smoke, exp_root):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments
    torch.manual_seed(seed)
    train_rows, held = _split(exp_root)
    print(f"[force] seed={seed}: {len(train_rows)} image-rows (modality-dropout "
          f"{MODE_P}) / {len(held)} held-out")
    model, processor, lm_path = load_qwen_omni_for_training()
    model.thinker = get_peft_model(model.thinker, build_lora_config(lm_path))
    assert_lora_applied(model.thinker, f"force-s{seed}")
    model.thinker.train()
    ds = DropoutDataset(train_rows.head(8) if smoke else train_rows, processor, exp_root, seed)
    out_dir = GPU_DIR / "track_d_force_ckpt" / tag(seed)
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
        print("[force] smoke done — inspect loss, then rerun without --smoke-test")
        return None, None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.thinker.save_pretrained(str(out_dir))
    print(f"[force] adapter saved to {out_dir}")
    return model, processor, held


def load_for_eval(seed):
    from peft import PeftModel
    base, processor, _ = load_qwen_omni_for_training()
    base.thinker = PeftModel.from_pretrained(base.thinker, str(GPU_DIR / "track_d_force_ckpt" / tag(seed)))
    _, held = _split(EXP_ROOT)
    return base, processor, held


def evaluate(seed, model, processor, held, exp_root):
    """Eval each held-out job under its image_condition, but with F0-contour
    images (swap the stored spectrogram path). image_wrong_audio keeps its wrong
    audio; wrong_image keeps a wrong (F0) image."""
    import numpy as np
    import torch
    model.eval()
    # existing F0 pngs (only the shortlist was rendered) -> wrong_image fallback
    f0_root = exp_root / "stimuli" / _DIRNAME[IMAGE_KIND]
    existing = {str(p.relative_to(exp_root)) for p in f0_root.rglob("*.png")}

    def f0_img(image_path, job_id):
        wav = image_path.replace("stimuli/spectrograms/", "stimuli/").replace(".png", ".wav")
        cand = _f0_for(wav)
        if cand in existing:
            return cand
        pool = sorted(existing)                       # wrong-image partner not in shortlist
        return pool[int(np.random.default_rng(abs(hash(job_id)) % (2**32)).integers(len(pool)))]

    results = []
    for row in held.itertuples():
        content = []
        if isinstance(row.audio_path, str):
            content.append({"type": "audio", "audio": str(exp_root / row.audio_path)})
        # remap image to the F0 contour of whatever the stored image referred to
        if isinstance(row.image_path, str):
            content.append({"type": "image", "image": str(exp_root / f0_img(row.image_path, row.job_id))})
        content.append({"type": "text", "text": row.prompt})
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
    print(f"[force] seed={seed}: {len(results)} responses -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--image-kind", choices=["f0contour", "f0zoom"], default="f0contour")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    a = ap.parse_args()
    IMAGE_KIND = a.image_kind
    exp_root = Path(a.exp_root)
    if a.eval_only:
        m, p, h = load_for_eval(a.seed); evaluate(a.seed, m, p, h, exp_root)
    else:
        m, p, h = train(a.seed, a.smoke_test, exp_root)
        if m is not None:
            evaluate(a.seed, m, p, h, exp_root)

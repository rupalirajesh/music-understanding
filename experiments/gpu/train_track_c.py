"""Track C (H100): 3-arm LoRA fine-tune on Audio-Flamingo-3, testing whether
the alignment-fixable shortlist (octave_id, tuning_judgment,
cents_discrimination, note_count -- L2-high/L3-low per the dissociation
table, PAPER.md / RESEARCH_PLAN.md §6) is actually fixable by fine-tuning,
and where.

  python gpu/train_track_c.py --arm llm_only    --smoke-test
  python gpu/train_track_c.py --arm llm_only
  python gpu/train_track_c.py --arm llm_encoder
  python gpu/train_track_c.py --arm control
  python gpu/train_track_c.py --eval-only --arm llm_only   # rerun eval only

ALWAYS run --smoke-test first per arm (trains ~8 steps on 8 examples, no
checkpoint saved) and eyeball the printed loss/generations before the full
run — same discipline as every other gpu/ script in this repo.

Three arms, same recipe, different LoRA target modules / training data:
  llm_only     LoRA on the LLM decoder only, encoder frozen. If this alone
               recovers the gap, it's an alignment problem (RQ3-b).
  llm_encoder  LoRA on LLM decoder + AF-Whisper encoder. If THIS closes it
               but llm_only doesn't, the encoder itself needed adjusting.
  control      Same LoRA recipe as llm_only, but trained on instrument_id
               instead of the shortlist -- a task that's already
               behaviorally solved by every model (Track A: 6/6, see
               PAPER.md). Controls for "any fine-tuning helps a little,"
               which would make llm_only's result meaningless on its own.

Held-out split: by soundfont (HELD_OUT_SOUNDFONTS below), not just held-out
clips -- guards against timbre leakage per RESEARCH_PLAN.md §0.6/§6.3.

Eval output: writes results/trackA/responses__af3-lora-<arm>.parquet in the
EXACT schema run_local.py produces (job_id, model, raw_response, error, ts),
so the existing scoring pipeline just works, zero new scoring code:
  python -m musicprobe.scoring --model af3-lora-llm_only
  python scripts/04_export_for_review.py --model af3-lora-llm_only
The pre-fine-tune baseline for comparison is already in
results/trackA/responses__nvidia_audio-flamingo-3-hf.parquet -- no need to
re-run it.

SCOPE NOTE (2026-07-24): trains on Battery v1's existing shortlist jobs
(~600 audio-condition jobs across the 4 tasks), NOT a dedicated larger v2
training pool -- RESEARCH_PLAN.md §6 calls for 5-20k QA pairs per skill
generated fresh; that's future work if this first pass shows signal.
Battery v1 stays FROZEN (decision 10, PROJECT_STATE.md) -- this script only
READS manifests/stimuli.parquet + jobs.parquet, never regenerates them.

UNVERIFIED on hardware, same honesty as every other gpu/ script:
- _find_language_model's candidate paths are a best guess (mirrors
  _find_audio_tower in extract_activations.py, whose audio_tower guess was
  later confirmed correct for AF3 -- but the language-model path has NOT
  been checked against the real "-hf" checkpoint). If it raises, run
  `for n, m in model.named_modules(): print(n, type(m).__name__)` on the
  loaded model and hardcode the real path into LM_PATH_CANDIDATES.
- LORA_TARGET_SUFFIXES assumes standard Qwen2-family attention projection
  names (q_proj/k_proj/v_proj/o_proj) for the LLM and standard Whisper
  attention names for the encoder. assert_lora_applied() below hard-fails
  if zero modules matched, instead of silently training nothing.
- build_lora_config() passes target_modules as a single regex STRING (peft
  matches it with re.search against each full module name) -- supported
  since peft ~0.5, but not version-pinned here; if get_peft_model() rejects
  a string, pin a peft version that supports it rather than switching to a
  plain suffix list (a suffix list can't distinguish the LLM's q_proj from
  the encoder's q_proj, which is the whole point of the regex).
- _build_example() assumes processor.tokenizer.eos_token exists (standard
  for a chat-template AutoProcessor, unconfirmed on this specific "-hf"
  checkpoint) -- if it's None, print(processor.tokenizer) and use whatever
  the real EOS string/id is.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musicprobe.config import EXP_ROOT, RESULTS_DIR, MANIFEST_PATH  # noqa: E402
from musicprobe.jobs import JOBS_PATH  # noqa: E402

GPU_DIR = Path(__file__).resolve().parent  # for checkpoint dirs: robust regardless of cwd,
# same reasoning as musicprobe.config's EXP_ROOT-based absolute paths -- a bare relative
# "gpu/track_c_checkpoints/..." only works if invoked with experiments/ as cwd, which every
# other gpu/ script avoids by importing absolute paths from musicprobe.config instead.

MODEL_NAME = "nvidia/audio-flamingo-3-hf"
SHORTLIST_TASKS = ["octave_id", "tuning_judgment", "cents_discrimination", "note_count"]
CONTROL_TASK = "instrument_id"          # already solved behaviorally (6/6 models) — arm (iii)
HELD_OUT_SOUNDFONTS = {"timgm"}         # excluded from training for every arm; eval-only
LM_PATH_CANDIDATES = ["language_model", "model.language_model", "model.model.language_model"]
AUDIO_PATH_CANDIDATES = ["audio_tower", "model.audio_tower"]  # verified 2026-07-24, commit 83c722c
LORA_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _find_submodule(model, candidate_paths, class_name_hints):
    """Same pattern as extract_activations.py's _find_audio_tower: try known
    paths first, fall back to scanning named_modules() for a class-name hint.
    Returns (path_string, submodule)."""
    for path in candidate_paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return path, obj
        except AttributeError:
            continue
    for name, mod in model.named_modules():
        if any(h in type(mod).__name__ for h in class_name_hints):
            return name, mod
    raise AttributeError(
        f"couldn't find a submodule matching {candidate_paths} or class hints "
        f"{class_name_hints}. Run `for n,m in model.named_modules(): "
        "print(n, type(m).__name__)` on the loaded model and hardcode the real "
        "path into this script's *_PATH_CANDIDATES.")


def load_af3_for_training():
    """AF3 in float32. bf16 trains fine but CRASHES at generation/eval with
    'Input type (float) and bias type (c10::BFloat16) should be the same' — the
    AF3 audio tower has mixed-precision internals that emit float32 during
    generate() (same reason run_local.py loads AF3 float32). float32 makes the
    train + eval path consistent AND matches the AF3 Track-A baseline this is
    compared against; fits easily on an 80GB+ GPU."""
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cuda")
    lm_path, _ = _find_submodule(model, LM_PATH_CANDIDATES,
                                 ("Qwen2ForCausalLM", "Qwen2Model", "CausalLM"))
    audio_path, _ = _find_submodule(model, AUDIO_PATH_CANDIDATES,
                                    ("AudioFlamingo3Encoder", "AudioTower", "WhisperEncoder"))
    print(f"[train_track_c] language model at '{lm_path}', audio tower at '{audio_path}'")
    return model, processor, lm_path, audio_path


def build_lora_config(arm: str, lm_path: str, audio_path: str):
    from peft import LoraConfig

    suffix_group = "|".join(LORA_TARGET_SUFFIXES)
    if arm in ("llm_only", "control"):
        pattern = rf"^{lm_path}\..*\.({suffix_group})$"
    elif arm == "llm_encoder":
        pattern = rf"^({lm_path}|{audio_path})\..*\.({suffix_group})$"
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM", target_modules=pattern)


def assert_lora_applied(model, arm: str):
    n = sum(1 for name, p in model.named_parameters() if p.requires_grad)
    assert n > 0, (
        f"arm={arm}: zero trainable parameters after get_peft_model — the "
        "target_modules regex matched nothing. Print named_modules() and fix "
        "LORA_TARGET_SUFFIXES / the discovered LM/audio paths before rerunning.")
    print(f"[train_track_c] arm={arm}: {n} trainable LoRA parameter tensors")


# ---------------------------------------------------------------- data -----
def _held_out_mask(sub: pd.DataFrame) -> pd.Series:
    """Held-out criterion, chosen per row from whichever factor carries
    leakage risk for that task: soundfont for instrument-rendered tasks
    (octave_id, note_count, instrument_id) -- guards against timbre leakage,
    same discipline as RESEARCH_PLAN.md §0.6/§6.3. tuning_judgment and
    cents_discrimination are pure-tone numpy synthesis with NO soundfont
    factor at all (verified 2026-07-24 -- a soundfont-only split would
    silently put 100% of them in training and leave eval empty), so those
    fall back to a held-out pitch register instead (top quintile of
    base_midi) -- guards against pitch-memorization leakage instead."""
    def parse(f):
        return json.loads(f) if isinstance(f, str) else {}
    factors = sub["factors"].apply(parse)
    has_soundfont = factors.apply(lambda d: "soundfont" in d)
    mask = pd.Series(False, index=sub.index)
    if has_soundfont.any():
        mask[has_soundfont] = factors[has_soundfont].apply(
            lambda d: d["soundfont"] in HELD_OUT_SOUNDFONTS)
    no_soundfont = ~has_soundfont
    if no_soundfont.any():
        midi = pd.to_numeric(factors[no_soundfont].apply(lambda d: d.get("base_midi")),
                             errors="coerce")
        threshold = midi.quantile(0.8)
        mask[no_soundfont] = midi >= threshold
    return mask


def _load_split(jobs_path: str, manifest_path: str, tasks: list[str]):
    jobs = pd.read_parquet(jobs_path)
    man = pd.read_parquet(manifest_path)[["stimulus_id", "factors"]]
    sub = jobs[(jobs["task"].isin(tasks)) & (jobs["condition"] == "audio")].merge(
        man, on="stimulus_id", how="left")
    held_out_mask = _held_out_mask(sub)
    held_out, train = sub[held_out_mask], sub[~held_out_mask]
    for t in tasks:
        n_ho = (held_out["task"] == t).sum()
        assert n_ho > 0, (
            f"task={t}: held-out split produced 0 rows -- the leakage-relevant "
            "factor for this task isn't soundfont or base_midi; check its "
            "factors and extend _held_out_mask before training on it.")
    return train.reset_index(drop=True), held_out.reset_index(drop=True)


def _build_example(processor, exp_root: Path, prompt: str, audio_path: str, answer: str):
    """Same conversation format as run_local.py's load_audio_flamingo /
    attention_audio.py's prepare_audio_flamingo (verified-working AF3 prompt
    format) -- tokenize prompt+answer together, mask labels on the prompt
    span so loss is computed on the answer tokens only (standard SFT)."""
    import torch

    content = [{"type": "audio", "path": str(exp_root / audio_path)},
               {"type": "text", "text": prompt}]
    conversation = [{"role": "user", "content": content}]
    prompt_inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    answer_ids = processor.tokenizer(answer + processor.tokenizer.eos_token,
                                      return_tensors="pt",
                                      add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prompt_inputs["input_ids"], answer_ids], dim=1)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    out = dict(prompt_inputs)
    out["input_ids"] = input_ids
    out["labels"] = labels
    if "attention_mask" in out:
        out["attention_mask"] = torch.cat(
            [out["attention_mask"], torch.ones_like(answer_ids)], dim=1)
    return out


class TrackCDataset:
    def __init__(self, rows, processor, exp_root):
        self.rows, self.processor, self.exp_root = rows, processor, exp_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return _build_example(self.processor, self.exp_root, r.prompt,
                              r.audio_path, r.ground_truth)


# ------------------------------------------------------------- training ----
def train(arm: str, smoke_test: bool, exp_root: Path, jobs_path: str, manifest_path: str):
    import torch
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments

    tasks = [CONTROL_TASK] if arm == "control" else SHORTLIST_TASKS
    train_rows, held_out_rows = _load_split(jobs_path, manifest_path, tasks)
    print(f"[train_track_c] arm={arm}: {len(train_rows)} train / "
          f"{len(held_out_rows)} held-out (soundfonts={HELD_OUT_SOUNDFONTS})")

    model, processor, lm_path, audio_path = load_af3_for_training()
    cfg = build_lora_config(arm if arm != "control" else "llm_only", lm_path, audio_path)
    model = get_peft_model(model, cfg)
    assert_lora_applied(model, arm)
    model.train()

    ds = TrackCDataset(train_rows.head(8) if smoke_test else train_rows, processor, exp_root)
    out_dir = GPU_DIR / "track_c_checkpoints" / arm
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1 if smoke_test else 8,
        num_train_epochs=1 if smoke_test else 3,
        max_steps=8 if smoke_test else -1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1 if smoke_test else 10,
        save_strategy="no" if smoke_test else "epoch",
        report_to=[],
        remove_unused_columns=False,  # TrackCDataset isn't a datasets.Dataset;
        # Trainer's column-pruning inspects .column_names, which this plain
        # class doesn't have. Usually a safe no-op either way, but disabling
        # it outright removes the failure mode instead of hoping it no-ops.
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda batch: batch[0])
    trainer.train()

    if smoke_test:
        print("[train_track_c] smoke test done — inspect the loss above, then "
              "rerun WITHOUT --smoke-test for the full arm.")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"[train_track_c] arm={arm}: adapter saved to {out_dir}")
    return model, processor, held_out_rows


# ------------------------------------------------------------------ eval ---
def evaluate(arm: str, model, processor, held_out_rows, exp_root: Path):
    import torch

    model.eval()
    results = []
    for row in held_out_rows.itertuples():
        content = [{"type": "audio", "path": str(exp_root / row.audio_path)},
                   {"type": "text", "text": row.prompt}]
        conversation = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        results.append({"job_id": row.job_id, "model": f"af3-lora-{arm}",
                        "raw_response": raw, "error": None, "ts": time.time()})

    out_path = RESULTS_DIR / f"responses__af3-lora-{arm}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"[train_track_c] arm={arm}: {len(results)} held-out responses -> {out_path}\n"
          f"  score with: python -m musicprobe.scoring --model af3-lora-{arm}\n"
          f"  compare against the pre-fine-tune baseline already in "
          f"results/trackA/responses__nvidia_audio-flamingo-3-hf.parquet")


def load_adapter_for_eval(arm: str):
    from peft import PeftModel

    base, processor, _, _ = load_af3_for_training()
    adapter_dir = str(GPU_DIR / "track_c_checkpoints" / arm)
    model = PeftModel.from_pretrained(base, adapter_dir)
    return model, processor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["llm_only", "llm_encoder", "control"])
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, load a previously-saved adapter and re-eval")
    ap.add_argument("--exp-root", default=str(EXP_ROOT))
    ap.add_argument("--jobs", default=str(JOBS_PATH))
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    if args.eval_only:
        tasks = [CONTROL_TASK] if args.arm == "control" else SHORTLIST_TASKS
        _, held_out = _load_split(args.jobs, args.manifest, tasks)
        model, processor = load_adapter_for_eval(args.arm)
        evaluate(args.arm, model, processor, held_out, exp_root)
    else:
        result = train(args.arm, args.smoke_test, exp_root, args.jobs, args.manifest)
        if result is not None:
            model, processor, held_out_rows = result
            evaluate(args.arm, model, processor, held_out_rows, exp_root)

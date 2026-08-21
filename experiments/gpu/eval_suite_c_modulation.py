"""Suite C, phenomenon 1 (modulation) -- the actual 5-condition grounding
test, run for real against the first item built by
scripts/generate_suite_c_modulation.py. Base model only (no fine-tuning --
Suite C tests whether a model can be honest about evidence when given it,
not whether fine-tuning helps yet).

Conditions (GROUNDING_PILOT_PLAN.md):
  1. audio_absent        -- text question only, no audio
  2. correct_audio       -- version A's audio (target G major)
  3. swapped_audio       -- version B's audio (target A minor) given for the
                             same question -- does the claim track the audio?
  4. audio_plus_tool     -- version A's audio + its own (correct) tool report
  5. audio_plus_wrong_tool -- version A's audio + version B's (wrong) tool report

  python gpu/eval_suite_c_modulation.py
"""
import json
import sys
import time
from pathlib import Path

import torch

GPU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU_DIR)); sys.path.insert(0, str(GPU_DIR.parent))
sys.path.insert(0, str(GPU_DIR.parent / "scripts"))
from musicprobe.config import EXP_ROOT, RESULTS_ROOT  # noqa: E402
from train_track_d import load_qwen_omni_for_training  # noqa: E402
from generate_suite_c_modulation import build_item  # noqa: E402

QUESTION = ("Does this song change key partway through? If so, where, and what's it "
            "moving to? Respond in exactly this format:\n"
            "claim: <your answer>\n"
            "evidence_span_seconds: <the timestamp range this claim rests on, or 'n/a'>\n"
            "measurement_or_tool_output: <the specific tool output you used, verbatim, "
            "or 'none used'>\n"
            "theory_rule_or_source_id: <the music-theory rule you applied, or 'n/a'>\n"
            "confidence: <low, medium, or high>")


def generate(processor, model, prompt, audio_path=None):
    content = []
    if audio_path is not None:
        content.append({"type": "audio", "audio": str(audio_path)})
    content.append({"type": "text", "text": prompt})
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, return_audio=False)
    if isinstance(out, tuple):
        out = out[0]
    out = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True)[0]


def run():
    item = build_item()
    audio_a = EXP_ROOT / item["audio_a"]
    audio_b = EXP_ROOT / item["audio_b"]

    model, processor, _ = load_qwen_omni_for_training()
    model.eval()

    conditions = {
        "audio_absent": dict(prompt=QUESTION, audio=None),
        "correct_audio": dict(prompt=QUESTION, audio=audio_a),
        "swapped_audio": dict(prompt=QUESTION, audio=audio_b),
        "audio_plus_tool": dict(
            prompt=f"{item['tool_report_a']}\n\n{QUESTION}", audio=audio_a),
        "audio_plus_wrong_tool": dict(
            prompt=f"{item['tool_report_b']}\n\n{QUESTION}", audio=audio_a),
    }

    results = []
    for cond_name, cfg in conditions.items():
        t0 = time.time()
        response = generate(processor, model, cfg["prompt"], cfg["audio"])
        elapsed = time.time() - t0
        results.append({
            "condition": cond_name, "prompt": cfg["prompt"],
            "audio": str(cfg["audio"]) if cfg["audio"] else None,
            "raw_response": response, "elapsed_s": round(elapsed, 2),
        })
        print(f"=== {cond_name} ({elapsed:.1f}s) ===")
        print(response)
        print()

    out_path = RESULTS_ROOT / "suite_c" / "modulation_mod_001_base_model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"item": item, "results": results}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    run()

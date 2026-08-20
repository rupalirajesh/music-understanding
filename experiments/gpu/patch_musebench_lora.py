"""Generate LoRA-wrapped copies of MUSE Benchmark's own Qwen2.5-Omni runner
scripts, for a real baseline-vs-fine-tuned comparison on MUSE (mentor's ask
2026-08-19 -- see BENCHMARK_LANDSCAPE.md Sec6 / PROJECT_STATE.md next action
26). The baseline half is already done for free (`parse_musebench_qwen.py`
mined MUSE's own already-published Qwen2.5-Omni logs) -- this generates the
fine-tuned half, so the two are directly comparable: same 10 tasks, same
prompts, same stimuli, same scoring, only the model's weights differ.

WHY PATCH THEIR SCRIPTS RATHER THAN REBUILD MUSE'S PROMPTS/PARSING/STIMULI-
GROUPING OURSELVES: these 10 runner scripts (~500 lines each, one per task)
are real, complete, already-verified-working code -- they're what generated
the exact logs `parse_musebench_qwen.py` already parsed for the baseline
number. Re-deriving the same prompts/parsing/scoring independently risks
subtle mismatches that would make "baseline vs fine-tuned" not actually
comparable. The only thing that needs to change for a fine-tuned run is
WHICH MODEL WEIGHTS get loaded -- so this patches exactly that, nothing else,
via a targeted regex substitution, not a manual rewrite of 10 files by hand.

WHAT THIS CHANGES, verified against all 10 real files 2026-08-19 (grepped
every file, not assumed uniform):
  1. `LOCAL_QWEN_DIR = "/scratch/bc3189/..."` (their own cluster's hardcoded
     absolute path, present verbatim in 9/10 files) -> an env-var-overridable
     default of `MODEL_ID` ("Qwen/Qwen2.5-Omni-7B"), so `local_files_only=
     True` resolves via the standard HF cache-by-repo-id lookup instead --
     this project's H100 box should already have that repo id cached from
     running the main battery, no download needed. `keymod_*_runner.py`
     already uses `os.environ.get(...)` for this (the one file that isn't
     uniform) -- left untouched, same env var name reused
     (`QWEN2_5_OMNI_LOCAL_DIR`) so one `export` covers all 10.
  2. The real (uncommented) `model = Qwen2_5OmniForConditionalGeneration
     .from_pretrained(...).eval()` block gets a PEFT wrap appended right
     after it -- same `.thinker`-only wrapping pattern as
     `image_track_common.py`/`attention_audio.py`/`eval_cmibench.py`.
     `keymod_*_runner.py` has a COMMENTED-OUT duplicate of this block (an
     older version left in as a comment) -- the regex anchors on
     start-of-line `model = ` with no `#` before it, so the commented block
     is correctly never matched (verified against this exact file).
  3. `make_log_filename`'s return f-string gets `_Qwen2.5-Omni_CHAT_` ->
     `_Qwen2.5-Omni-LORA-<tag>_CHAT_` so fine-tuned logs never collide with
     or overwrite the original baseline logs already sitting in
     `Gemini_Qwen_AF_logs/`.

USAGE:
  export QWEN2_5_OMNI_LOCAL_DIR=Qwen/Qwen2.5-Omni-7B   # or a local snapshot path
  python gpu/patch_musebench_lora.py --muse-dir <muse_dir> \\
      --lora-checkpoint <path to a Track E or D-zoom checkpoint> --tag e_f0text \\
      --out-dir <muse_dir>/Qwen2.5-Omni-LORA
  cd <muse_dir>/Qwen2.5-Omni-LORA && python chord_quality_Qwen2.5-Omni_runner.py
  # ... one per task, same as MUSE's own README's "how to run" section
  python gpu/parse_musebench_qwen.py --muse-dir <muse_dir> \\
      --log-dir <muse_dir>/Qwen2.5-Omni-LORA/logs --tag-filter LORA-e_f0text
  # (parse_musebench_qwen.py needs --log-dir/--tag-filter added to read these
  # -- not done yet, small follow-up once real logs exist to test the regex
  # against; flagging rather than guessing the exact output-dir convention
  # these patched scripts will actually write logs to)

STATUS 2026-08-19: regex patterns built and checked against the real text of
all 10 files (string search, shown above) but the FULL patch has only been
dry-run (`--dry-run`, prints match count per file, writes nothing) -- not
yet applied and executed against a real checkpoint (no GPU on the laptop).
Run `--dry-run` first on the H100 box and confirm "1 from_pretrained block
patched, 1 log-filename line patched" per file before trusting a real run.
"""
import argparse
import re
from pathlib import Path

FROM_PRETRAINED_RE = re.compile(
    r'^(?P<indent>[ \t]*)model = Qwen2_5OmniForConditionalGeneration\.from_pretrained\(\n'
    r'(?:.*\n)*?'
    r'(?P=indent)\)\.eval\(\)\n',
    re.MULTILINE,
)
LOCAL_DIR_RE = re.compile(
    r'^(?P<indent>[ \t]*)LOCAL_QWEN_DIR = "/scratch/bc3189/[^"]*"\n',
    re.MULTILINE,
)
LOG_TAG_RE = re.compile(r'_Qwen2\.5-Omni_CHAT_')

PEFT_WRAP_TEMPLATE = (
    "{indent}from peft import PeftModel\n"
    "{indent}model.thinker = PeftModel.from_pretrained(model.thinker, {ckpt!r})\n"
    "{indent}print('[patch_musebench_lora] wrapped .thinker with adapter from ' + {ckpt!r})\n"
)


def patch_one(text: str, lora_checkpoint: str, tag: str) -> tuple[str, dict]:
    counts = {"local_dir": 0, "from_pretrained": 0, "log_tag": 0}

    def _local_dir_sub(m):
        counts["local_dir"] += 1
        indent = m.group("indent")
        return f'{indent}LOCAL_QWEN_DIR = __import__("os").environ.get("QWEN2_5_OMNI_LOCAL_DIR", MODEL_ID)\n'
    text = LOCAL_DIR_RE.sub(_local_dir_sub, text, count=1)

    def _fp_sub(m):
        counts["from_pretrained"] += 1
        indent = m.group("indent")
        return m.group(0) + PEFT_WRAP_TEMPLATE.format(indent=indent, ckpt=lora_checkpoint)
    text = FROM_PRETRAINED_RE.sub(_fp_sub, text, count=1)

    def _log_sub(m):
        counts["log_tag"] += 1
        return f"_Qwen2.5-Omni-LORA-{tag}_CHAT_"
    text = LOG_TAG_RE.sub(_log_sub, text)

    return text, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muse-dir", required=True, type=Path)
    ap.add_argument("--lora-checkpoint", required=True,
                     help="path to a saved PEFT adapter dir (e.g. a Track E f0text or "
                          "Track D-zoom checkpoint) -- baked into the patched scripts as a "
                          "literal string, not read at runtime")
    ap.add_argument("--tag", required=True, help="short label for log filenames, e.g. 'e_f0text'")
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="default: <muse-dir>/Qwen2.5-Omni-LORA-<tag>")
    ap.add_argument("--dry-run", action="store_true",
                     help="print match counts per file, write nothing -- run this first")
    args = ap.parse_args()

    src_dir = args.muse_dir / "Qwen2.5-Omni"
    out_dir = args.out_dir or (args.muse_dir / f"Qwen2.5-Omni-LORA-{args.tag}")
    files = sorted(src_dir.glob("*_Qwen2.5-Omni_runner.py"))
    if not files:
        raise SystemExit(f"no *_Qwen2.5-Omni_runner.py files under {src_dir} -- wrong --muse-dir?")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        text = f.read_text()
        patched, counts = patch_one(text, args.lora_checkpoint, args.tag)
        flag = "OK" if counts["from_pretrained"] == 1 and counts["log_tag"] >= 1 else "CHECK ME"
        print(f"{f.name}: local_dir={counts['local_dir']} from_pretrained={counts['from_pretrained']} "
              f"log_tag={counts['log_tag']}  [{flag}]")
        if not args.dry_run:
            (out_dir / f.name).write_text(patched)

    if args.dry_run:
        print("\n--dry-run: nothing written. Every row should read "
              "'from_pretrained=1 log_tag>=1 [OK]' -- if any row says [CHECK ME], "
              "read that file's model-loading block directly before trusting a real patch.")
    else:
        print(f"\nwrote {len(files)} patched scripts to {out_dir}")


if __name__ == "__main__":
    main()

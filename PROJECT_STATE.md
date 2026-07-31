# PROJECT STATE — the everything doc (pick-up-where-we-left-off)

Last updated: 2026-07-29. Update this whenever anything changes hands.

## Machines & workflow
- **Laptop** (no GPU): stimulus synthesis, API evals, scoring/analysis,
  stimulus auditing by ear, manual response verification.
- **H100 box**: open-model runs, activation extraction/probing, generation
  battery; outputs are committed back to `results/` and pushed.
- Repo: https://github.com/rupalirajesh/music-understanding (PUBLIC — keep
  docs free of personal details).

## What exists (all pushed, commit a6ae042 + later)
- `RESEARCH_PLAN.md` — full background primer + study design (L1/L2/L3,
  tracks A/B/C, hypotheses H1–H6). Read Part 0 first.
- `experiments/TASKS.md` — the tier list: every task, N, controls, what a
  pass/fail means, sample-size rationale.
- `experiments/musicprobe/` — package: generators, manifest, prompts (3
  paraphrases + diagnostic distractors), jobs (controls), runners (API +
  H100 local), scoring (accuracy/audio_gain/confusions/psychometrics), DSP
  L1 baselines.
- `experiments/manifests/` — FROZEN battery v1: 1,176 stimuli / 2,063 jobs.
  Do not regenerate while v1 runs are in flight; changes = v2.
- `experiments/gpu/` — Track B: activation extraction (MERT/Whisper/CLAP
  loaders done; AF3/Qwen-Omni TODO) + held-out-soundfont linear probes.
- `experiments/genmodel/` — MusicGen constraint battery + DSP adherence
  scoring; single-constraint rule; vocab-vs-theory pairs. Not yet run.
- `experiments/scripts/` — 00 soundfonts, 01 generate (seeded, machine-
  stable), 02 jobs, 03 listening page, 04 review export, 05 self-test
  (14 checks, ALL PASSING as of 2026-07-18).
- Docs workflow: a local read-this inbox file (not in git, cleared
  iteratively), `PAPER.md` (results skeleton), this file (state).

## Status of runs (updated 2026-07-22)
| What | Where | Status |
|---|---|---|
| Qwen2-Audio-7B full battery | H100 box | done |
| Qwen2.5-Omni-7B, Qwen3-Omni-30B, AF3, Music-Flamingo full battery | H100 box | done (landed commit 4aa5dcf) |
| Gemini-2.5-Pro battery (via Portkey) | H100/laptop → API | done |
| GPT-4o-audio | — | **out of scope — no OpenAI API access** (removed from the plan 2026-07-25) |
| MOSS-Music-8B-Instruct (new, added 2026-07-25, Sethu's initiative) | H100 box | full Track A battery done |
| MERT/Whisper/CLAP extraction + probes | H100 box | done — all 3 encoders × 11 tasks |
| Own-encoder re-probe (key_id/mode_id/chord_quality/interval_id) | H100 box | done 2026-07-24, commit 83c722c |
| Attention diagnostic (`gpu/attention_audio.py`) | H100 box | **DONE CORRECTLY 2026-07-25** (commit c348ea6, eager-attention verified, all 5 models, `--per-task 6`) — supersedes the 2026-07-24 retracted run. See Known gaps for the corrected finding. |
| Microtone probe (relative-pitch direction + absolute detune, new task, Sethu's initiative) | H100 box | done 2026-07-25, commit b0dec99 |
| Track C (3-arm LoRA on AF3: llm_only / llm_encoder / control) | H100 box | **done 2026-07-26** (commit `a37cc8a`) — `octave_id`/`note_count` alignment-fixable (+0.50/+0.59 and +0.43/+0.40 over baseline); `tuning_judgment`/`cents_discrimination` resist fine-tuning even with the encoder also tuned |
| Track D Phase 1 (Qwen2.5-Omni-7B + spectrogram image, single LoRA arm) | H100 box | done 2026-07-26 (commit `a37cc8a`) — first-pass "image helps cents" (0.67→0.77) turned out to be an OOD training confound; see conclusive rerun below |
| Track D conclusive (mixed-condition training, paired 3-seed McNemar) | H100 box | done 2026-07-28 (commit `d7d4c3f`) — precise null: spectrogram adds nothing on any task (Δacc 95%CI includes 0 everywhere); mechanism controls confirm the model ignores it |
| Track D force (same spectrogram + modality-dropout training) | H100 box | done 2026-07-28 (commit `841da35`) — forces the model to actually use the image (wrong-image now craters perf, p=.003/.016) but accuracy still doesn't improve; spectrogram lacks fine cents-level detail |
| Track D zoom (zoomed F0-contour image + in-tune reference line) | H100 box | done 2026-07-29 (commit `03c7bde`) — cents 0.55→0.94 (p<1e-4), tuning 0.53→0.89 (p<1e-4); first method that fixes absolute tuning |
| Track E (pitch-tracker output as text in the prompt, audio-only) | H100 box | done 2026-07-29 (commit `03c7bde`) — cents 0.62→0.92 (p<1e-4), scalable/deployable (no image needed); tuning ns (text has no reference point) |
| Track F (learned pitch-stream fusion: trainable projector injects F0 features into embedding space) | H100 box | done 2026-07-29 (commit `2418e80`) — null; injection verified to reach the model (logit shift scales with pitch value) but behaviorally ignored — likely a data-size limit (~348 training examples to learn a new modality interface from scratch) |
| MusicGen battery | H100 box | done — tempo/key/register scored; meter/mode deliberately left for manual scoring |
| Track G (chromagram front-end for key_id/mode_id/chord_quality/interval_id — first causal test on the harmonic cluster; Tracks C-F only ever targeted pitch/tuning) | H100 box | **set up 2026-07-31, not yet run.** CPU groundwork done + committed: 664 chromagram PNGs rendered (whole battery, for a valid wrong-image draw pool), `manifests/chroma_jobs.parquet` built (1760 rows), held-out split sanity-checked (287 train / 612 held-out job rows, non-empty for all 4 tasks). `scripts/15_run_track_g.sh` — smoke-test first. |
| Track H (in-audio reference tone for tuning_judgment — tests whether Track D-zoom's "needs an explicit reference" finding works delivered in-AUDIO instead of switching modality) | H100 box | **set up 2026-07-31, not yet run.** CPU groundwork done + committed: 120 stimuli x 2 new WAV variants (reftone/wrong_reftone) rendered, `manifests/reftone_jobs.parquet` built (360 rows), held-out split sanity-checked (93 train / 27 held-out stimuli, no overlap). `scripts/16_run_track_h.sh` — smoke-test first. |
| AF3 / Music Flamingo loaders | code | done, verified working |
| Qwen2-Audio published-number replication | — | still TODO; pick exact benchmark subset first |
| Analysis pass on all of the above | laptop | first pass done 2026-07-22 — see PAPER.md Results; dashboard + plots in `experiments/results/trackB/analysis/`; attention + microtone graphs in `experiments/results/trackB/attention/attention_graph.png` and `.../probes/microtone_probe_graph.png` (2026-07-25) |

## Decisions made (and why) — chronological
1. Only fine-tuning is feasible (no pretraining budget); the study's job is
   to find WHERE fine-tuning helps (alignment/data gaps) vs can't (encoder).
2. Test generation models too — may end up fine-tuning one. Understanding
   measured as constraint adherence scored by our own DSP.
3. Synthetic stimuli for Tiers 1–2 (unlimited N, perfect ground truth,
   factor control); real datasets only for Tier 3.
4. Meter task = beats-per-bar (audio can't convey notated denominator —
   caught in stimulus audit). 3/4-vs-6/8 grouping task specced separately.
5. Key task "cadence" form renamed "progression" (it's not cadence-type ID);
   cadence_type specced as future task 2.6.
6. tuning_judgment added (12-TET grid probe, behavioral half).
7. All responses stored verbatim; response-level verification is done
   manually (review CSVs + explanations.csv). Never trust auto-parse alone.
8. Single-constraint generation prompts only; compound later.
9. WAVs/soundfonts not in git; regeneration is deterministic (stable crc32
   seeding — NEVER use builtin hash() for seeds).
10. Battery v1 frozen at 2,063 jobs. Ladder arm will be battery v2.
11. NEW DIRECTION (2026-07-18): representation ladder → universal
    music representation → world-model framing. 5 requirements: inferable
    from audio / token-compact / sufficient for battery / human-readable /
    genre-universal incl. continuous pitch (staff notation fails #5; that's
    the Carnatic module's future role as stress test).
12. Tracks C–F (2026-07-26 to 07-29): the alignment-gap hypothesis (H4) is only half
    right for pitch. LoRA on the existing audio→LM pathway (Track C) fixes readout-level
    gaps (`octave_id`, `note_count`) but not microtone perception, even with the encoder
    also tuned. Microtone perception turned out to be fixable, but only by routing it
    through a representation the model already knows how to read — pitch-tracker output
    as **text** (Track E: fixes relative pitch, no image needed, cheap/deployable) or a
    **zoomed chart with an explicit reference line** (Track D-zoom: the only method that
    also fixes absolute tuning). An end-to-end learned fusion into embedding space
    (Track F) was a clean null at this data scale (~348 examples) — bolting on a raw
    feature and hoping a small adapter learns to use it does not work; reusing an
    existing, pretrained interface (numbers, images) does. Treat this as the working
    default for any future modality-injection experiment, not just pitch.
13. Tracks G/H (2026-07-31, set up, not yet run): two more front-end candidates in the
    same spirit as C-F, chosen to extend into a cluster and a delivery mechanism neither
    C-F touched. Track G asks whether the "give it a rendered chart" idea (Track D)
    generalizes past pitch to the harmonic cluster (`key_id`/`mode_id`/`chord_quality`/
    `interval_id`) via a chromagram (12 pitch-class rows x time — the harmonic analogue of
    the F0-contour). Track H asks whether Track D-zoom's "needs an explicit reference"
    finding for absolute tuning can be delivered in-AUDIO (mix a reference tone into the
    clip itself) rather than switching modality to vision at all — cheaper than rendering
    an image if it works, and a genuinely different test of the same underlying claim.
    Both reuse the established discipline from the start (dropout-style training so eval
    conditions stay in-distribution, wrong-condition mechanism controls, held-out splits,
    paired McNemar over 3 seeds) rather than repeating Track D Phase 1's single-arm mistake.

## Next actions (ordered, updated 2026-07-31)
GPT-4o-audio removed from this list entirely 2026-07-25 — no OpenAI API access, out of
scope. 6 models (Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni-30B, AF3, Music-Flamingo,
Gemini-2.5-Pro) + MOSS-Music-8B is the Track A roster unless that changes.

1. ~~Re-run the attention diagnostic on the 4 newer open models~~ **DONE 2026-07-25**
   (commit c348ea6, eager-verified). See Known gaps above for the corrected findings.
2. ~~Sanity-check `beats_per_bar` and `mode_id` by hand~~ **DONE 2026-07-31** — no scoring/
   parsing bug (both use the same generic open-format substring match and the same 10%
   `WRONG_AUDIO_FRACTION` design as every other task, verified against `jobs.py`). Re-derived
   the wrong-audio deltas directly from `scored__*.parquet`: beats_per_bar's "inversion" is
   real in direction (mean +0.156 across 7 models, 5/7 positive) but its control is only
   n=14/model (std 0.141) — about 1 SE, not the clean signal PAPER.md previously implied.
   mode_id is weaker still (+0.070, std 0.169) — within noise, no action needed. Revised
   PAPER.md language to "low-confidence, directionally suggestive," not "structurally
   broken." Before any future LoRA arm on beats_per_bar, enlarge the wrong-audio sample
   first (bump `WRONG_AUDIO_FRACTION` for this task or pool paraphrases) — n=14 isn't
   enough to certify or refute the task.
3. ~~Re-probe each LALM's OWN encoder~~ **DONE 2026-07-24** (commit 83c722c,
   `gpu/extract_activations.py --own-encoder`, submodule paths verified:
   `thinker.audio_tower` for Qwen-Omni, `model.audio_tower` for Flamingo). Result: own-encoder
   probe accuracy on `key_id`/`mode_id`/`chord_quality`/`interval_id` stays modest —
   comparable to, not clearly beating, the generic MERT/Whisper/CLAP baselines. Reading
   unchanged from before the re-probe: behavioral success on these 4 tasks is more likely
   priors than a richer internal representation. No further action.
4. Run the L1 DSP floor (`musicprobe.l1_baselines`, or essentia/madmom for a stronger
   key/beat detector) on the same stimuli so every task sits on the full L1→L2→L3 ladder.
   **Scoped precisely 2026-07-31**: `l1_baselines.run()` currently only covers 4/13 tasks
   (`pitch_note_id`, `cents_discrimination`, `tempo_bpm`, `key_id`) — `beats_per_bar`,
   `note_count`, `octave_id`, `tuning_judgment`, `instrument_id`, `interval_id`,
   `chord_quality`, `mode_id`, `progression_id` have **no L1 baseline at all** yet, not just
   a weak one. Notably `beats_per_bar` (the task flagged in item 2 above) has never had an
   L1 check — an essentia `RhythmExtractor2013` beat-tracking floor on the same stimuli
   would directly test whether the ground-truth beat/meter labels are even DSP-recoverable,
   which would settle whether the wrong-audio anomaly is a task-difficulty issue or a label
   issue. Tried `pip install essentia` in the laptop venv (2026-07-31): **fails to build**
   (no prebuilt wheel for this Python/platform combo, source build errors in setuptools) —
   confirms the existing plan's call to do this on the H100/Linux box, not the laptop.
   Did not attempt writing essentia-based key/beat/chord detectors blind (9 tasks, no local
   way to verify correctness) — this is real DSP-implementation work best done iteratively
   where essentia actually installs, not guessed from the laptop.
5. Qwen2-Audio published-number replication (harness validation). **Benchmark decision made
   2026-07-31**: use MuChoMusic (ISMIR'24, arxiv 2408.01337, 1.1K validated MCQs), not
   MMAU-music — MuChoMusic is a single well-defined public set (vs MMAU-music being one
   subscore inside the larger multi-domain MMAU benchmark, arxiv 2410.19168, with less
   clarity on exact subset reconstruction), and it's the benchmark this project's own
   text-prior/no-audio-control methodology is explicitly modeled after (RESEARCH_PLAN.md
   §0.6), making it the more meaningful fidelity check. Could NOT confirm Qwen2-Audio's
   exact published MuChoMusic number via web search with confidence — search results
   surfaced a 51.4% figure for Qwen-**Audio** (the v1 predecessor), not Qwen2-Audio, and a
   0.692 MMAU-Music figure for Qwen2.5-**Omni**, neither the right model. Pull the real
   number directly from the MuChoMusic paper's model-comparison table (arxiv 2408.01337) or
   the Qwen2-Audio technical report (arxiv 2407.10759) before running the replication —
   don't trust a secondhand number. Still needs the actual eval run (H100/API).
6. Ladder arm (battery v2) — L1 features into prompts at one-abstraction-below-answer,
   features-only + few-shot variants; keep v1 job_ids untouched. Deferred behind 1–5.
   Note (2026-07-29): Tracks D-zoom/E are effectively a one-task preview of this arm
   (in-context F0 feature, text and image forms) — worth reusing their harness rather
   than rebuilding from scratch when this starts.
7. ~~Track C is set up, not yet run~~ **DONE 2026-07-26** (commit `a37cc8a`) — see Status
   table. `octave_id`/`note_count` alignment-fixable; `tuning_judgment`/`cents_discrimination`
   resist LoRA even with the encoder tuned — carried forward into Track D/E/F below.
8. ~~Track D Phase 1 is set up, not yet run~~ **DONE, then superseded 3x, 2026-07-26 to
   07-29** — Phase 1's apparent win was an OOD confound (commit `d7d4c3f` conclusive
   rerun: true null). Forcing image use didn't help either (commit `841da35`: mechanism
   works, accuracy doesn't). A zoomed F0-contour + reference-line image finally fixed
   both cents and absolute tuning (commit `03c7bde`, Track D-zoom). Parallel text-only
   front-end (Track E, same commit) fixes cents without an image but not tuning. An
   end-to-end learned fusion alternative (Track F, commit `2418e80`) was a null — the
   model ignores a raw feature injected into embedding space, likely too little data
   (~348 examples) to learn a new modality interface from scratch. See Status table and
   PAPER.md Results for full numbers.
9. ~~Wire up before/after L2 probes on the Track D-zoom / E checkpoints~~ **RESOLVED
    ANALYTICALLY 2026-07-31, no GPU run needed** — `build_lora_config()` (shared by
    train_track_d/_force/_conclusive, train_track_e_f0text, train_track_f_pitchfuse) only
    matches `target_modules` under `thinker.<lm_path>` (the LLM decoder); the regex never
    reaches `audio_tower` or the vision tower, so those receive zero gradient in every
    Track D/E/F run — the encoder's own representation is provably unchanged (bit-identical
    forward pass pre/post fine-tune). Re-running `extract_activations.py --own-encoder`
    against these checkpoints would just reproduce Track B's existing numbers. Confirms the
    fix is entirely LLM-decoder/read-side, consistent with the substitution-not-hearing
    mechanism check. Written up in RESEARCH_PLAN.md §12.6.
10. ~~Decide + write up the pitch-representation recommendation~~ **DONE 2026-07-31** —
    RESEARCH_PLAN.md §12.6 (new section) walks decision 11's 5 requirements against the
    actual Track C–F results and states the recommendation (F0-as-text for relative pitch,
    reference-anchored image for absolute tuning, no raw learned fusion at this data scale).
11. **New (2026-07-31)**: Track F aug's leakage bug is now fixed in code — `generate_aug.py`
    was sampling `base_midi` from the same 52–76 range as the frozen battery, which overlaps
    the held-out eval band (base_midi ≥ ~71.13, per `train_track_c.py`'s quantile split); it
    now caps training pitches at `MAX_TRAIN_MIDI = 70` (a 1-semitone margin below the
    threshold, since cents stimuli detune up to 100¢). Regenerated locally on the laptop
    (`scripts/generate_aug.py` + `musicprobe.pitch_feats --manifest aug_train_jobs.parquet` —
    both pure CPU/numpy/librosa, no GPU needed) and both committed. **Still needs a GPU
    rerun**: `python gpu/train_track_f_pitchfuse.py --aug` (3 seeds) to get a clean,
    unconfounded read on whether the audio-only baseline's jump (cents 0.62→0.89 in the
    leaky run) survives once training pitches can no longer leak into the held-out band.
    The fusion-null verdict itself doesn't need re-running — it was a same-model paired
    comparison, unaffected by this bug.
12. **New (2026-07-31)**: Tracks G (chromagram, harmonic cluster) and H (in-audio
    reference tone, tuning_judgment) are set up and ready for the H100 GPU steps —
    `scripts/15_run_track_g.sh` / `scripts/16_run_track_h.sh`. CPU-side groundwork (stimulus/
    image rendering, job-hygiene layers, held-out splits) is done, committed, and verified
    locally: Track G's 1760-row `chroma_jobs.parquet` gives 287 train / 612 held-out job
    rows (non-empty across all 4 tasks); Track H's 360-row `reftone_jobs.parquet` gives 93
    train / 27 held-out stimuli with zero train/held overlap. Both scripts smoke-test first,
    same discipline as every other `gpu/` LoRA script — the model-loading/PEFT paths are
    reused verbatim from Track D/C (`load_qwen_omni_for_training`, `build_lora_config`,
    `_held_out_mask`), so they inherit those scripts' hardware verification, but the new
    dataset-construction code (chromagram rendering, reftone synthesis, job hygiene) is
    unverified beyond the CPU-side checks already run.

## Known gaps / honesty list
- L1 key detection weak on minor progressions (naive Krumhansl) — use
  essentia on GPU box before making L1 claims about cadence-form stimuli.
- Mode melodies are random diatonic walks — musician-composed melodies
  would be better stimuli (planned).
- Real Gemini/OpenAI payload paths and Qwen loading only testable on the
  GPU box (self-test covers everything else).
- Explain-format responses can be confabulated — evidence, not proof.
- progression_id has only 32 stimuli (small N — hypothesis-generating only).
- Contamination: synthetic stimuli are safe, but Tier 3 datasets are public.
- **key_id MCQ distractors are circle-of-fifths neighbors + the relative key by design**
  (`musicprobe/prompts.py`) — MCQ confusion matrices for this task will always look
  "musically structured" regardless of whether the model is really listening. Only the
  open-format subset is a valid listening-structure signal, and n=20/model there is small.
- MCQ vs open-format accuracy gaps are large and inconsistent across models/tasks (not just
  a minor scoring nuance) — e.g. Gemini scores 0/20 open-format on `key_id` despite ~42%
  MCQ; `progression_id` is 0% open-format for 4/6 models. Any task/model accuracy number
  should be read alongside its MCQ-vs-open split before being quoted, not just the pooled
  accuracy.
- `beats_per_bar`'s audio_gain is negative for 4/6 models AND its wrong-audio-control delta
  is negative on average (models do *better* with swapped-in wrong audio than the correct
  clip) — the only task where this happens. Treat this as a task-validity flag, not a
  capability finding, until it's manually audited (next action #3).
- ~~The Track B attention diagnostic on the 4 newer models is RETRACTED~~ **RESOLVED
  2026-07-25** (commit c348ea6): re-run with `assert_eager_attention()` passing for every
  model (top-level + every sub-config resolved to `'eager'` — the 2026-07-24 sdpa-fallback
  risk did not recur). Corrected findings, which **supersede** the retracted run and are now
  safe to cite:
  - No decay across generation steps — the retracted run's "listens first, then coasts" was
    itself an artifact of the unverified run, not a real pattern.
  - Every model attends to audio tokens *below* the uniform baseline (audio is ~55–67% of
    context; actual attention is 0.03–0.31) — all models systematically under-weight audio
    relative to its share of the input, to very different degrees.
  - Real cross-architecture variation, confirmed this time: Qwen2-Audio highest (~0.31,
    structured peak in early-mid layers), Music-Flamingo/AF3 ~0.08, Qwen2.5-Omni ~0.045,
    Qwen3-Omni-30B lowest (~0.03). Qualitatively similar ordering to the retracted run, but
    now trustworthy — graph at
    `results/trackB/attention/attention_graph.png` (`gpu/plot_attention.py`).
- No-audio "refusal" behavior is not comparable across models as a single number: Gemini
  refuses explicitly (57%), Qwen2-Audio refuses some (13%), but AF3/Music-Flamingo/
  Qwen2.5-Omni/Qwen3-Omni never refuse (0%) — they answer confidently and wrong instead,
  which is a worse failure mode than refusing, not a better one, even though 0% refusal
  reads as "more helpful" at a glance.

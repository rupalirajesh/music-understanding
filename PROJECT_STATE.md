# PROJECT STATE — the everything doc (pick-up-where-we-left-off)

Last updated: 2026-07-22. Update this whenever anything changes hands.

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
| GPT-4o-audio | — | still not run — **the one remaining Track A cell** |
| MOSS-Music-8B-Instruct (new, added 2026-07-25, Sethu's initiative) | H100 box | full Track A battery done |
| MERT/Whisper/CLAP extraction + probes | H100 box | done — all 3 encoders × 11 tasks |
| Own-encoder re-probe (key_id/mode_id/chord_quality/interval_id) | H100 box | done 2026-07-24, commit 83c722c |
| Attention diagnostic (`gpu/attention_audio.py`) | H100 box | **DONE CORRECTLY 2026-07-25** (commit c348ea6, eager-attention verified, all 5 models, `--per-task 6`) — supersedes the 2026-07-24 retracted run. See Known gaps for the corrected finding. |
| Microtone probe (relative-pitch direction + absolute detune, new task, Sethu's initiative) | H100 box | done 2026-07-25, commit b0dec99 |
| Track C (3-arm LoRA on AF3) | H100 box | **set up, not yet run** — `scripts/09_run_track_c.sh` |
| Track D Phase 1 (Qwen2.5-Omni-7B + spectrogram-image) | H100 box | **set up, not yet run** — `scripts/11_run_track_d.sh` |
| MusicGen battery | H100 box | done — tempo/key/register scored; meter/mode deliberately left for manual scoring |
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

## Next actions (ordered, updated 2026-07-25)
1. **GPT-4o-audio run** (Sethu, H100/API) — the one remaining Track A cell. Still not done —
   oldest item on this list, keeps getting passed over for other work.
2. ~~Re-run the attention diagnostic on the 4 newer open models~~ **DONE 2026-07-25**
   (commit c348ea6, eager-verified). See Known gaps above for the corrected findings.
3. **Sanity-check `beats_per_bar` and `mode_id` by hand** before trusting any model
   comparison on them — 4/6 models show negative audio_gain and `beats_per_bar` shows an
   *inverted* wrong-audio-control result (worse with correct audio than swapped audio).
   Read ~20 raw responses per model in `results/trackA/review__<model>/beats_per_bar.csv`
   and `mode_id.csv` to rule out a scoring/parsing bug before concluding it's a real
   model failure or task-design flaw. **Last remaining blocker on the Track C shortlist**
   (item 6) — everything else it depends on is now resolved. Still not done — this one is
   manual (read raw responses by hand), not a script Sethu can just run.
4. ~~Re-probe each LALM's OWN encoder~~ **DONE 2026-07-24** (commit 83c722c,
   `gpu/extract_activations.py --own-encoder`, submodule paths verified:
   `thinker.audio_tower` for Qwen-Omni, `model.audio_tower` for Flamingo). Result: own-encoder
   probe accuracy on `key_id`/`mode_id`/`chord_quality`/`interval_id` stays modest —
   comparable to, not clearly beating, the generic MERT/Whisper/CLAP baselines. Reading
   unchanged from before the re-probe: behavioral success on these 4 tasks is more likely
   priors than a richer internal representation. No further action.
5. Run the L1 DSP floor (`musicprobe.l1_baselines`, or essentia/madmom for a stronger
   key/beat detector) on the same stimuli so every task sits on the full L1→L2→L3 ladder.
6. Qwen2-Audio published-number replication (harness validation) — still not done.
7. Ladder arm (battery v2) — L1 features into prompts at one-abstraction-below-answer,
   features-only + few-shot variants; keep v1 job_ids untouched. Deferred behind 1–6.
8. **Track C is set up, not yet run** (2026-07-24, `experiments/scripts/09_run_track_c.sh` +
   `experiments/gpu/train_track_c.py`) — 3-arm LoRA fine-tune on AF3, shortlist
   `octave_id`/`tuning_judgment`/`cents_discrimination`/`note_count` (`beats_per_bar`
   provisionally excluded pending item 3, doesn't block starting on the other four). Sethu:
   pull and run `bash scripts/09_run_track_c.sh` — smoke-tests each arm first.
9. **Track D Phase 1 is set up, not yet run** — full design in
   RESEARCH_PLAN.md §12. Targets **Qwen2.5-Omni-7B only** — model-support audit (§12.1)
   found AF3 and Music-Flamingo don't accept image input at all, so Track C's target model
   can't run this.
   - Groundwork done + verified locally 2026-07-24/25 (no GPU needed): spectrograms rendered
     for all 1,101 available stimuli (`scripts/10_render_spectrograms.py`), image hygiene
     layer built and checked by hand (`musicprobe/image_jobs.py` →
     `manifests/image_jobs.parquet`, 1,416 jobs, 4-task shortlist).
   - Training + eval script written 2026-07-25 (`gpu/train_track_d.py`) — single LoRA arm
     (Thinker language-model decoder only) trained on the `image` condition, evaluated on
     held-out stimuli across all three image_conditions, scored via a small wrapper around
     `musicprobe.scoring.parse_response/is_correct` (image_condition isn't part of the
     existing scoring pipeline's condition grouping). Data-pipeline half tested locally
     (held-out split, scoring aggregation — caught and fixed a real bug where unparseable
     answers were being dropped from the denominator instead of scored incorrect, same
     class of issue `scoring.py:107-108` already documents). **GPU-bound half (model
     loading, LoRA application, actual training) is UNVERIFIED on hardware** — smoke-test
     first (`--smoke-test`), same discipline as Track C; see the script's docstring for the
     one thing most likely to need a hardcoded fix (the Thinker's language-model submodule
     path).
   - Before/after L2 probes (does joint training change the audio encoder itself) not yet
     wired up — reuse `extract_activations.py --own-encoder` after a successful full run,
     not automated in `train_track_d.py` yet.
   - Sequenced behind Track C's own-encoder work only in the sense that they reuse the same
     LoRA stack; not actually blocked on Track C finishing.
   - Sethu: pull and run `bash scripts/11_run_track_d.sh` — smoke-tests first, renders any
     missing spectrograms before training (idempotent, skips existing PNGs).

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

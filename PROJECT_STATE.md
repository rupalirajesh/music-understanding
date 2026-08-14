# PROJECT STATE — the everything doc (pick-up-where-we-left-off)

Last updated: 2026-08-05. Update this whenever anything changes hands.

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
| Track G (chromagram front-end for key_id/mode_id/chord_quality/interval_id — first causal test on the harmonic cluster; Tracks C-F only ever targeted pitch/tuning) | H100 box | **done 2026-08-05** (commit `8050378`, Sethu) — null on all 4 tasks (3-seed paired McNemar): key_id Δ=−0.07 (p=.26), mode_id Δ=+0.04 (p=.57), chord_quality Δ=+0.13 (p=.11, n=72), interval_id Δ=+0.03 (p=.64); all CIs include 0. Mechanism controls: wrong-chromagram ≈ no-chromagram everywhere (content not read); wrong-audio+chromagram only craters key_id (p=.02). Note: this tested ONE variant — a force-style-trained but flat, unannotated, unzoomed chromagram — the analogue of Track D's early "conclusive/force" stage, not the full iteration that eventually found D-zoom. The zoomed/annotated step that actually rescued pitch has not been tried for harmony yet; see next action 13. |
| Track H (in-audio reference tone for tuning_judgment — tests whether Track D-zoom's "needs an explicit reference" finding works delivered in-AUDIO instead of switching modality) | H100 box | **done 2026-08-05** (commit `8050378`, Sethu) — flat null: reftone vs plain Δ=+0.01 (p=1.0); wrong_reftone vs plain Δ=+0.02 (p=.82) — a WRONG reference doesn't mislead any more than a correct one, so the model isn't comparing target-to-reference in audio at all. D-zoom's reference fix is visual-channel-specific, not a general "give it a reference" effect. |
| Track F-aug leakage fix (rerun of the 9x-data pitch-fusion aug run) | H100 box | **done 2026-08-05** (commits `8050378`+`b54c4bf`, Sethu) — `generate_aug.py` was sampling training pitches into the held-out eval band; capped at `MAX_TRAIN_MIDI=70` and reran. Corrected audio-only-on-held-out: cents 0.62→0.78→**0.74** (clean, vs 0.89 leaky) — about half the leaky jump was real. tuning 0.51→0.68→**0.62** (clean, vs 0.83 leaky) — nearly halved, now ≈ 2AFC majority rate, NOT evidence of learned absolute-tuning perception. Fusion-null verdict unchanged (model still ignores the fused stream). |
| L1 DSP floor extension (10/13 tasks, up from 4/13) | laptop | **done 2026-08-05** — octave_id 1.00, cents/pitch_note_id 1.00 (existing), tempo_bpm 0.82 (existing), key_id 0.80 (existing), chord_quality 0.60, tuning_judgment 0.63, interval_id 0.54, note_count 0.40, **mode_id 0.25** (weakest — now confirmed hardest task in the battery by 3 independent methods: L1/L2/L3). `beats_per_bar`/`progression_id`/`instrument_id` still need essentia, H100/Linux box only. |
| Tracks L-Q (harmony representation ladder: peak-picked chroma → zoomed peak-picked chroma → multi-pitch line graph → zoomed line graph → piano-roll → tonal centroid, key_id/mode_id/chord_quality/interval_id) | H100 box | **done 2026-08-06** (commits `bf83b07` seed-0, `6eef62a` full 3-seed, Sethu) — clean null on `mode_id`/`interval_id` across all 6 representations; **`key_id` significantly HURT by every representation** (Δ −0.075 to −0.117, worst tonnetz p=.016, N p=.043, L p=.029); `chord_quality` trends positive on all 6 (+0.08 to +0.17, piano-roll best) but underpowered (p=.043 only for piano-roll, n=72). Mechanism: wrong_image≈no_image (model mostly ignores the image), but image+wrong_audio hurts key_id (p=.02) — not fully inert. Headline: the D-zoom trick does not transfer to harmony as tested. See decision 16 — L-Q tested zoom and reference/richness as *separate* ingredients; the combination (Track X) was still untried as of this run. |
| Tracks R-W (rhythm representation ladder: tempogram → peak-picked tempogram → onset-strength line graph → zoomed line graph → rhythm-roll → rhythm necklace, tempo_bpm/beats_per_bar) | H100 box | **done 2026-08-06** (commits `bf83b07` seed-0, `6eef62a` full 3-seed, Sethu) — first causal fine-tuning of any kind on this cluster. **`tempo_bpm` significantly HURT by 4/6 representations** (tempogram p=.002, peak-tempogram p=.006, onset-line p=.039, rhythm-roll p=.039; zoom-onset/necklace ns but still negative in direction, no representation reverses it); `beats_per_bar` null everywhere (n=33, underpowered). Same headline as L-Q: no representation helps, several hurt. See decision 16 — same "ingredients tested separately" gap; Track Y is the untried combination. |
| Tracks X/Y (the missing zoom+explicit-reference combination for harmony/rhythm: X = zoomed peak-picked chroma + estimated-tonic reference row; Y = zoomed rhythm-roll) | H100 box | **CPU-side groundwork done + verified 2026-08-06, GPU steps not yet run.** Both renderers built, run against the full 1248-stimulus battery (0 errors each). Held-out splits verified to exactly match their parent ladder (X: 287/612 = Track G/L-Q's split; Y: 127/132 = Track R-W's split), 0 overlap, all tasks present both sides. Registered in `gpu/train_track_repr.py`'s `TRACKS` dict (`X`, `Y`), `gpu/analyze_track_repr.py` picks them up automatically (registry-driven, no changes needed there). `scripts/19_run_tracks_xy.sh` + `scripts/RUNBOOK_tracks_xy.md` — smoke-test first. |
| AF3 / Music Flamingo loaders | code | done, verified working |
| Qwen2-Audio / MuChoMusic replication | — | **resolved 2026-08-05, deprioritized** — confirmed via primary source the MuChoMusic paper only evaluates Qwen-Audio v1, never Qwen2-Audio, so there's no published number to replicate against for the actual Track A roster model. Scaffold at `gpu/eval_muchomusic.py`, not queued as active work. |
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
13. Tracks G/H (2026-08-05, run and landed): two more front-end candidates in the same
    spirit as C-F, chosen to extend into a cluster and a delivery mechanism neither C-F
    touched. Track G asked whether the "give it a rendered chart" idea (Track D)
    generalizes past pitch to the harmonic cluster (`key_id`/`mode_id`/`chord_quality`/
    `interval_id`) via a chromagram (12 pitch-class rows x time — the harmonic analogue of
    the F0-contour). Track H asked whether Track D-zoom's "needs an explicit reference"
    finding for absolute tuning can be delivered in-AUDIO (mix a reference tone into the
    clip itself) rather than switching modality to vision. **Both came back null** — see
    Status table for numbers. Reading: the D-zoom reference-line fix does not generalize
    into "any chart helps" or "any reference helps" — it's specific to an annotated visual
    position for pitch. Harmony (key/mode/chord/interval) remains an open gap with no
    working front-end tried yet. Both runs reused the established discipline (dropout-style
    training so eval conditions stay in-distribution, wrong-condition mechanism controls,
    held-out splits, paired McNemar over 3 seeds) rather than repeating Track D Phase 1's
    single-arm mistake, so the null results are trustworthy, not a methodology artifact.

## Next actions (ordered, updated 2026-08-12)
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
4. ~~Run the L1 DSP floor~~ **PARTIALLY DONE 2026-08-05** (laptop, pure numpy/scipy, no
   essentia needed — `musicprobe/l1_baselines.py`) — extended from 4/13 to 10/13 tasks.
   New estimators: `octave_estimate` (reuses `f0_autocorr`), `tuning_estimate` (nearest-
   12-TET-grid distance, 25¢ threshold), `note_count_estimate`/`interval_estimate`
   (FFT peak-picking with greedy harmonic-series rejection), `chord_quality_estimate`/
   `mode_estimate` (extend `key_estimate`'s Krumhansl-correlation method to CHORDS'/MODES'
   binary templates x 12 roots/tonics). Actual run, `python -m musicprobe.l1_baselines`:
   | task | L1 acc | n | chance | note |
   |---|---|---|---|---|
   | octave_id | **1.00** | 72 | — | trivial from audio; matches Track C's LoRA-fixable verdict |
   | cents_discrimination | 1.00 | 180 | — | (existing) |
   | pitch_note_id | 1.00 | 72 | — | (existing) |
   | tempo_bpm | 0.82 | 60 | — | (existing) |
   | key_id | 0.80 | 96 | ~4% (24-way) | (existing) |
   | chord_quality | 0.60 | 96 | 12.5% (8-way) | strong recoverable signal |
   | tuning_judgment | 0.63 | 120 | 50% | above chance but noisy — naive threshold, not tuned |
   | interval_id | 0.54 | 144 | ~8% (12-way) | recoverable, well above chance |
   | note_count | 0.40 | 100 | ~20-30%ish | noisy heuristic, real headroom, not a hard ceiling |
   | mode_id | **0.25** | 104 | ~8% (13-way) | weakest of the six — barely 3x chance, and this now lines up with L2 (own-encoder probe also barely-above-chance for mode_id) AND L3 (behavioral ~chance) — three independent methods agree mode_id is the hardest task in the battery. Corroborates the existing known-gap note that mode melodies are random diatonic walks, not musician-composed — may be a stimulus-quality issue as much as a model one. |
   Still not covered: `beats_per_bar`, `progression_id` (need real beat/chord-sequence
   tracking, autocorrelation-only isn't trustworthy as a floor), `instrument_id` (already
   near-ceiling behaviorally, L1 floor isn't the interesting question there) — still
   genuinely blocked on essentia/madmom on the H100/Linux box (confirmed 2026-07-31:
   `pip install essentia` has no wheel for the laptop's Python/platform).
5. ~~Qwen2-Audio published-number replication~~ **RESOLVED 2026-08-05, still needs the GPU
   run**: confirmed via primary source (fetched the actual MuChoMusic paper, arxiv
   2408.01337, Table 3 — not a secondhand summary) that it evaluates **Qwen-Audio v1**
   (arxiv 2311.07919) at 51.4% overall / 51.1% knowledge / 51.0% reasoning / 89.7% IFR.
   **It does not evaluate Qwen2-Audio at all** — checked a 2025 follow-up (arxiv 2504.00369)
   that references Qwen2-Audio too; it only cites v1's number secondhand, no standalone v2
   result anywhere. So there is no published Qwen2-Audio MuChoMusic number to replicate —
   the original next-action framing was chasing a number that doesn't exist. Correct fix:
   point the harness at **Qwen-Audio v1** (`Qwen/Qwen-Audio-Chat` on HF) instead, since
   that's the only model with a citable ground truth for this benchmark; this is a one-off
   harness sanity check, not a Track A roster change (Qwen2-Audio stays as-is in Track A).
   Scaffold written: `gpu/eval_muchomusic.py` (dataset `mulab-mir/muchomusic` via HF
   `datasets`). **Unverified** — `datasets`/`huggingface_hub` aren't in the laptop venv, so
   this hasn't executed end to end; field names are a best guess from the dataset card, not
   confirmed against the real schema.
   **Deprioritized 2026-08-05 (Rupali's call)**: not worth running as originally scoped.
   The only way this buys real harness-validation value is against Qwen-Audio v1
   specifically (the one model with a citable number) — running it against Qwen2-Audio,
   Qwen3-Omni, or any other roster model would produce a number with nothing to check it
   against, i.e. not actually a validation. Script is left in place in case a v1 run is
   ever wanted as a cheap one-off sanity check; not queued as active work.
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
11. ~~Track F aug's leakage bug~~ **DONE 2026-08-05** (commits `8050378`+`b54c4bf`, Sethu) —
    `generate_aug.py` now caps training pitches at `MAX_TRAIN_MIDI = 70`, reran 3 clean
    seeds. Corrected audio-only-on-held-out: cents 0.74 (clean, was 0.89 leaky), tuning
    0.62 (clean, was 0.83 leaky) — see Status table. First commit (`8050378`) still had a
    stale leaky seed-2 response file (analyzed before that seed's clean eval finished);
    `b54c4bf` fixed it. Fusion-null verdict unchanged.
12. ~~Tracks G/H set up, not yet run~~ **DONE 2026-08-05** (commit `8050378`, Sethu) — both
    null. See Status table for numbers and PAPER.md Results/Conclusions for the full
    writeup. Narrows the D-zoom reference-line finding to "visual + pitch-specific,"
    not a general reference-giving or chart-giving principle.
13. **Corrected 2026-08-05 (was wrong in the first pass of this entry)**: the L2 own-encoder
    re-probe for `key_id`/`mode_id`/`chord_quality`/`interval_id` was already done on
    2026-07-24 (next action 3, `results/trackB/probes/probe__*_own__{task}__*.csv`) — no
    need to re-run it. Reading the actual numbers (best layer, full label-space, not the
    4-way MCQ subset): `key_id` best 0.17–0.23 vs chance 0.042 (~5x, the strongest signal
    of the four); `chord_quality` best 0.18–0.31 vs chance 0.125 (~2x); `interval_id` best
    0.16–0.18 vs chance 0.083 (~2x); `mode_id` best 0.04–0.12 vs chance 0.077 (barely above
    chance across every encoder — the weakest signal, closest to a true floor case). So
    3 of the 4 tasks have modest-but-real recoverable signal (not near-ceiling like
    `instrument_id`'s 94/91%, but not absent either) — `mode_id` is the one where a
    front-end may genuinely have nothing to work with.
    **Superseded 2026-08-05** — replaced with a fixed sequential pipeline (Rupali's call):
    peak-picked chroma → zoomed peak-picked chroma → line graph → zoomed line graph →
    piano-roll → tonal centroid, same discipline as C–F throughout (dropout training,
    wrong-condition mechanism controls, held-out splits, 3-seed paired McNemar). **Policy
    updated 2026-08-05 (second call)**: run the FULL sequence rather than stopping at the
    first win — the goal now includes "what works best / creates the richest internal
    representation," a comparison question the stop-early rule can't answer. Train each in
    order (still sequential so later steps can build on what earlier ones show), analyze
    all six once landed. Tracked as tasks (see task list, Tracks L/M/N/O/P/Q):
    - **Track L — peak-picked (binarized) chroma**: threshold Track G's `chroma_cqt` output
      (top-K active bins per frame → bright block, rest dark) instead of the raw continuous
      energy heatmap. Cheapest fix: overtones/timbre bleed energy into neighboring bins in
      the raw chroma, making it genuinely blurry even for clean notes; this removes that
      noise without needing full note transcription.
    - **Track M — zoomed peak-picked chroma**: same binarized chroma, higher resolution /
      stretched time axis — isolates the "zoom" half of D-zoom that Track G's flat chroma
      never tested (Track G already covered the "force" half; see corrected note above).
    - **Track N — line graph (multi-pitch trajectory)**: generalizes the F0-contour that
      worked for pitch to polyphonic content — audio-derived (not MIDI-derived) pitch
      trajectories per detected note, multiple simultaneous lines during a chord. Needs a
      feasibility check first: block-chord stimuli (near-simultaneous onsets) may not suit
      a "trajectory" framing as well as more sequential content (mode/interval melodic
      forms) — may need per-task handling.
    - **Track O — zoomed line graph**: same representation, zoomed, mirroring D-zoom's
      exact recipe generalized to multiple lines.
    - **Track P — piano-roll**: absolute pitch height × time, one block per detected note
      (duration included, audio-derived onset+pitch detection). Directly targets the
      simultaneity bottleneck TASKS.md already flagged for `chord_quality`/`interval_id`
      (arpeggiated-succeeds/block-fails) — chords show as vertically-stacked blocks at one
      time-coordinate, which neither chroma (sums into 12 bins) nor a single line can show.
      Last step in the sequence; if this is also null, the input-representation search for
      harmony is exhausted for now — the next lever is the auxiliary self-transcription
      training objective (§12.3), not another front-end.
    Explicitly ruled out this round (Rupali's calls, not just mine): a text-based reference
    hint (would repeat Track E's already-diagnosed substitution-not-hearing pattern without
    teaching the model to listen better, and the "give it a reference" idea already failed
    once in-audio via Track H); resynthesizing the input audio to remove complexity (not a
    deployable fix — users' music can't be simplified before asking the question); an
    MCQ-template-glyph image (assumes MCQ framing, which conflicts with the longer-term goal
    of open-ended, non-MCQ questions).
    CPU-side groundwork (rendering, manifests, held-out splits) can be done on the laptop,
    same as Track G/H; GPU training/eval steps still need the H100 box.
14. **New (2026-08-05)**: analogous representation sequence for the RHYTHM cluster
    (`tempo_bpm`, `beats_per_bar` — the two rhythm tasks in the frozen v1 battery;
    `grouping_3v6`, TASKS.md 2.7, is speced but not yet built). Unlike harmony, this
    cluster has had **zero** causal fine-tuning of any kind before now — no LoRA-only arm,
    no front-end. Same six-step ladder, mapped onto the rhythm-appropriate DSP analogues,
    same discipline as C–P (dropout training, wrong-condition mechanism controls, held-out
    splits, 3-seed paired McNemar), run in full (not stop-early, same policy as harmony's
    second call above). Tracked as tasks (Tracks R/S/T/U/V/W):
    - **Track R — tempogram** (chroma equivalent): `librosa.feature.tempogram` /
      `fourier_tempogram`, a periodicity-vs-time heatmap, same non-leakage rule as chroma
      (fixed BPM/lag axis, not tied to this stimulus's actual tempo/meter label).
    - **Track S — peak-picked tempogram**: threshold to the top-K periodicity peaks per
      frame, same fix as Track L applied to Track R.
    - **Track T — onset-strength line graph** (F0-contour equivalent): a single onset-
      envelope curve over time (`librosa.onset.onset_strength`) — literally the rhythm
      analogue of the pitch-contour line graph that worked for pitch.
    - **Track U — zoomed onset-strength line graph**: same, higher temporal resolution.
    - **Track V — beat/onset grid ("rhythm-roll", piano-roll equivalent)**: onset markers
      plotted against a metrical grid inferred from the audio's own detected tempo
      (subdivision lines from the SAME estimate the model would have access to, not from
      the ground-truth beats-per-bar label) — shows precisely when onsets land relative to
      a pulse grid without pre-supposing how many beats divide it.
    - **Track W — rhythm necklace / circular polygon** (tonal-centroid equivalent):
      audio-derived onset times folded modulo one estimated cycle length, plotted as dots
      on a circle (Toussaint, *The Geometry of Musical Rhythm* — "rhythm necklace"
      representations; onset patterns as convex polygons on a circle reveal evenness/
      symmetry properties). **Leakage care needed**: the circle's circumference must come
      from a detected periodicity (audio-derived), not from the ground-truth beats-per-bar
      count, or the number of dots the model sees would hand over the answer directly —
      same discipline as Track V's grid.
    CPU-side groundwork can start on the laptop now (all of librosa's tempogram/onset/
    onset_strength functions are already available in the venv, same as the tonnetz
    prototype above); GPU steps need the H100 box.
    **Built + tested 2026-08-05**: all 6 renderers written (`scripts/render_rhythm_repr.py`),
    verified on real stimuli, 3 real bugs caught and fixed during testing — `_reject_harmonics`-
    equivalent issue N/A here, but (1) the necklace's cycle-length scorer originally used
    unweighted onset timing, which can't distinguish n=3 from n=6 on a perfectly regular
    click train (a uniform train folded mod ANY integer multiple looks equally concentrated)
    — fixed by weighting the circular-concentration statistic by onset strength, so real
    accents (not just regular clicks) determine the cycle length; (2) the period estimator
    (envelope autocorrelation) regularly locked onto a 2x/3x sub-harmonic of the true click
    rate — replaced with the median inter-onset interval from librosa's own onset detector,
    which is far more direct and doesn't have the sub-harmonic failure mode; (3) a
    `numpy.ptp()` API break (removed from ndarray in this numpy version). Verified 12/15
    (80%) correct cycle-length detection across all 5 beats_per_bar categories (3/4/5/6/7)
    after the fixes. **Known limitation** (found by independent leakage-review pass,
    2026-08-05): Tracks V/W's `wrong_image` mechanism control draws from the whole battery
    (`image_jobs.py`'s uniform-draw design), and ~28% of a random battery sample (56/200)
    produce a near-blank rhythm-roll/necklace image (fewer than 2 onsets detected — mostly
    single sustained-tone stimuli from unrelated tasks, not click-based). This does **not**
    affect the primary image-vs-no_image analysis (verified: 0/160 of the rhythm cluster's
    OWN stimuli are sparse, they're all click tracks) — only dilutes the wrong_image
    control's statistical power for these two tracks specifically. Not fixed; flagging so
    Track V/W's wrong-image numbers get read with that caveat, not trusted at face value.
15a. **CRITICAL BUG found + fixed 2026-08-05, before ever reaching the GPU box** (independent
    correctness-review pass, confirmed separately by a direct local run of `split()` against
    the real rhythm-task jobs): `gpu/image_track_common.py` originally reused
    `train_track_c._held_out_mask` unchanged for every track's held-out split, same as
    Track G did. That function's fallback chain is soundfont -> top-quintile-`base_midi`;
    `tempo_bpm`/`beats_per_bar`'s factors are `{bpm, meter, n_bars[, beats]}` -- **neither**
    key present, so the fallback's `base_midi >= quantile(0.8)` comparison is `NaN >= NaN`,
    False for every row. Result: `_held_out_mask` silently returned all-False for Tracks
    R-W -- 0 rows held out, meaning training would have run on 100% of the rhythm data and
    `evaluate()` would write a 0-row `responses__*.parquet`, which `analyze_track_repr.py`
    would then crash reading back (`KeyError` on missing columns). This would only have
    surfaced AFTER a full remote GPU training run for all 6 rhythm tracks -- an expensive
    way to discover a data-splitting bug. Confirmed via direct test: `split()` on the real
    jobs returned `held=0` for every one of R/S/T/U/V/W before the fix.
    **Fix**: added a third fallback tier in `gpu/image_track_common._held_out_mask` (NOT
    modifying `train_track_c._held_out_mask` itself -- that function is already used by
    Track C/D/G/H's already-run results; changing it risks changing their behavior on any
    rerun) -- rows with neither soundfont nor base_midi now fall back to a held-out
    top-quintile-BPM split, same "hold out the tail of a continuous factor" discipline as
    the base_midi tier, guarding against tempo-memorization instead of pitch-memorization.
    Re-verified after the fix: all 12 tracks (L-Q AND R-W) now produce non-empty,
    non-overlapping train/held splits with every target task represented in both (L-O:
    287 train / 612 held, matching Track G's already-published split exactly; R-W: 127
    train / 132 held).
15b. **Second bug found + fixed 2026-08-05** (same independent review): `_peak_pick`
    (Tracks L/M, `scripts/render_harmony_repr.py`) and the tempogram-picking logic (Track S,
    `scripts/render_rhythm_repr.py`) used plain `argsort`-based top-k selection with no
    energy floor -- confirmed via testing that a fully silent frame (all-zero chroma/
    tempogram) still gets exactly k bins marked "on" (argsort of an all-zero/tied array
    still returns k indices). Any stimulus with lead-in/trailing silence or a rest gap would
    get fabricated "active pitch class"/"active periodicity" markings on frames with no real
    content. Fixed: both now gate on frame energy (silent/near-silent frames, <2% of the
    stimulus's peak frame energy, are left fully off) before picking top-k on the rest.
16. **Tracks L-Q/R-W full 3-seed results landed (2026-08-06, commit `6eef62a`, Sethu)
    + Tracks X/Y set up in response**: the 12-representation sweep is a clean null —
    `mode_id`/`interval_id`/`beats_per_bar` null throughout, `key_id` and `tempo_bpm`
    actively HURT by every/most representations, `chord_quality` suggestive but
    underpowered. On reviewing why pitch (Track D-zoom) succeeded where this ladder
    didn't: D-zoom wasn't just "zoom" (Track D force = resolution/forced-attention
    alone, null) and wasn't just "add a reference" (Track H = in-audio reference
    alone, null) — it was zoom AND an explicit annotated reference position
    *together*; neither ingredient alone fixed pitch either, across 3 earlier
    attempts. L-Q/R-W tested zoom (M/O/U) and reference/richness (P/V) as *separate*
    items on a 6-item list, the same way D-force and Track H each isolated one
    ingredient — nobody had tried the combination on harmony/rhythm. Built Tracks
    X (zoomed peak-picked chroma + estimated-tonic reference row, tonic estimated
    via Krumhansl correlation on the stimulus's own chroma — same non-leakage
    discipline as D-zoom's pyin-estimated reference) and Y (zoomed rhythm-roll —
    Track V's detected-pulse-grid chart at Track U's finer time resolution) to fill
    that specific gap, not as two more independent guesses. CPU-side groundwork done
    + verified (see Status table); GPU run not yet done (no GPU on the laptop).
    `scripts/19_run_tracks_xy.sh` + `scripts/RUNBOOK_tracks_xy.md`.
17. **New (2026-08-05)**: auxiliary self-transcription training objective (RESEARCH_PLAN.md
    §12.3, specced since before Track D but never run) — extend to run across **all three**
    clusters (pitch, harmony, rhythm), not harmony alone. This is the one intervention in
    the whole project where the model generates its own intermediate representation during
    training rather than reading an externally-injected one — the framing every front-end
    track above is really scaffolding toward, not a replacement for Tracks L–Y.
    **Format blocker resolved 2026-08-12**: JSON event list of audio-derived
    `(onset, dur, hz)` triples — see RESEARCH_PLAN.md §12.3 for the full decision and why
    MIDI-as-text/prose/pitch-bend formats were rejected. CPU-side groundwork done + verified
    (laptop): `musicprobe/transcription_target.py` (reuses Track P's onset+piptrack detection
    chain, duplicated not imported, so Track P's landed results can't be affected), run
    against the full 1248-stimulus battery, 0 errors, cached to
    `manifests/transcription_target.json` (1104 unique audio_paths, median target 197 chars,
    dense-rhythm tail up to ~4.1k chars — flag for capping if that's too long once training
    starts). **Multi-task LoRA trainer built 2026-08-12** — `gpu/train_track_z_transcribe.py`
    (Track Z), mirrors `train_track_e_f0text.py`'s dropout-dataset pattern: each training step
    is randomly `answer` (weight 0.6, the normal battery question) or `transcribe` (weight 0.4,
    a fixed prompt against the JSON target above), same audio clip either way. At eval time
    only `answer` is used (per §12.3 — the transcribe head is discarded, only its effect on the
    shared encoder is being tested). Runs across all 13 tasks, reusing
    `image_track_common._held_out_mask`'s 3-tier fallback (the only held-out function in this
    codebase confirmed to generalize past harmony/rhythm). **CPU-side verified 2026-08-12**:
    `split()` produces non-empty train/held rows for every one of the 13 tasks (1170 train /
    517 held total; per-task train/held ranges from `progression_id` 30/16 to `beats_per_bar`
    126/18); every training row's `audio_path` has a transcription-target entry (0 missing,
    checked directly). **GPU steps unverified** (no GPU on the laptop, same status as every
    other track before its first real run) — `train()`/`evaluate()` are written and
    syntax-checked, not executed. **Still needed after a checkpoint exists**: the before/after
    L2 probe comparison itself isn't automated — re-run `gpu/extract_activations.py
    --own-encoder` against the Track Z checkpoint, then `gpu/probe.py` (and/or
    `gpu/probe_mlp.py`, next action 19) against the pre-fine-tune baseline;
    `evaluate()` prints this reminder at the end of its own run.
18. **New (2026-08-12)**: check what Qwen2-Audio/Qwen2.5-Omni were actually trained and
    evaluated on for music, before reading any more nulls as purely a representation-choice
    finding — **done as desk research 2026-08-12**. Qwen2-Audio's own published benchmark
    roster (Qwen team blog + technical report, arxiv 2407.10759): LibriSpeech/Common
    Voice/Fleurs/Aishell2 (ASR), CoVoST2 (speech translation), Meld (speech emotion),
    Vocalsound (non-speech human vocalizations — cough/laugh/sneeze), and AIR-Bench (a
    GPT-4-graded chat benchmark with 4 dimensions: speech/sound/music/mixed). Music is **not**
    a first-party target anywhere in that roster — it's one of four sub-dimensions inside one
    broader benchmark, not a dedicated eval. Qwen2.5-Omni's roster (Qwen team blog): Common
    Voice (ASR), CoVoST2 (translation), MMAU (audio understanding), OmniBench, Seed-tts-eval
    — again no dedicated music benchmark; music enters only via MMAU, a **third-party**
    benchmark (arxiv 2410.19168) spanning Speech/Sound/Music domains, on which Qwen2-Audio
    scores 55.4% overall (same "third-party, not the model's own claimed number" status as
    MuChoMusic's Qwen-Audio-v1 number in next action 5). **Reading**: neither model's own
    training/eval pipeline treats music as a first-class target — it rides along inside
    broader "audio understanding" benchmarks. This doesn't mean music is absent from
    pretraining data (no technical report gives a data-mix breakdown by content type; PDF
    table extraction failed locally, not re-attempted), but it does mean the L-Q/R-W nulls
    and actively-negative results on `key_id`/`tempo_bpm` are consistent with "this model was
    never optimized to be good at this," not only "we haven't found the right front-end yet"
    — worth stating explicitly in PAPER.md's limitations/interpretation, not just here.
    **Not done**: pulling exact AIR-Bench music-dimension numbers (image/table-only in the
    PDF, couldn't extract without `poppler`/OCR tooling not installed on the laptop) — if a
    precise number matters later, install `poppler` (`brew install poppler`) and re-run
    `Read` with `pages` on the cached PDF, or source the number from a paper that quotes it
    in text (same "primary source over secondhand summary" discipline as next action 5).
    **Superseded/expanded 2026-08-13**: full comprehensive version now in
    `BENCHMARK_LANDSCAPE.md` §3 (per prof's explicit request for comprehensiveness) —
    extends this same desk-research discipline across the whole model roster (Music
    Flamingo, Audio Flamingo 3, Qwen3-Omni, Qwen3.5-Omni, SALMONN, Gemini native audio,
    MU-LLaMA/MusiLingo/M2UGen), each claim cited to an arXiv ID (8 spot-checked against
    `export.arxiv.org`, all resolved correctly). Headline: general-purpose LALMs (Qwen
    family, Gemini) name zero music-specific training datasets anywhere — music enters only
    as an eval target; only Music Flamingo/AF3 and the frozen-MERT music specialists
    (MU-LLaMA/MusiLingo/M2UGen) have legible music-specific training data. Qwen2-Audio's
    Figure 3 data-mix chart is confirmed still unresolved (image-only, retried and failed
    again) — a genuine dead end short of manually reading the PDF visually. Also found a
    newer Qwen3.5-Omni report (arXiv 2604.15804, ~Apr 2026) that explicitly states it does
    NOT disclose a music/speech/sound content-type breakdown — useful as a citable
    "the field doesn't report this," not a continuing unknown to chase further.
    `BENCHMARK_LANDSCAPE.md` §1-2 also separately answers "what datasets exist to test on"
    and "how do published models score on them" (30+ datasets/benchmarks surveyed,
    per-model published scores with own-paper vs. third-party provenance flagged), and §4
    is the benchmark-desiderata analysis (why none of these substitute for our own battery).
19. **New (2026-08-12)**: train a nonlinear decoder (small MLP, not `LogisticRegression`) at
    each encoder layer, reusing activations already extracted for Track B (MERT/Whisper/CLAP
    + each LALM's own encoder) — every existing probe in `gpu/probe.py`/`probe_microtone.py`/
    `probe_vision_pitch.py` is linear-only. Motivation: the near-floor tasks (`mode_id` best
    0.04–0.12 vs chance 0.077, `interval_id`/`chord_quality` modest-but-real per next action
    13) were certified "weak signal" by a linear probe specifically — a nonlinear decoder
    could recover structure a hyperplane can't separate, which would change the read on
    whether these tasks are genuinely information-poor at the encoder or just
    linearly-inseparable. No new data collection (activations already on the H100 box from
    Track B); this is a new analysis script, `sklearn.neural_network.MLPClassifier` or a
    small torch head, same train/held split discipline as `probe.py`. Cheap relative to any
    LoRA track — CPU-feasible if the saved activation `.npy`/`.parquet` files are small
    enough to pull to the laptop, otherwise a quick H100 job.
    **Built 2026-08-12**: `gpu/probe_mlp.py`, drop-in second pass over the same `--acts`
    directory `probe.py` reads (identical `load_xy`, reused not copied), `MLPClassifier`
    (one hidden layer, 32 units, `early_stopping=True` — deliberately small given ~100-200
    examples/fold, so it catches simple nonlinear separability without just memorizing).
    Output lands in the same `results/trackB/probes/` dir as `probe__*.csv`, named
    `probe_mlp__*.csv` for a direct diff. **Smoke-tested against synthetic fake activations**
    (no real `.npz` files exist on the laptop — `musicprobe/config.py`: "activations stay on
    the GPU box"): built a fake encoder dir with a deliberately-nonlinear (XOR-like) toy
    target, ran end-to-end, 0 crashes, MLP scored 0.433 vs 0.333 chance — confirms the
    plumbing (load → fold-split → fit → score → write) works, but this is NOT a real result;
    run against an actual `--acts` directory on the H100 box before trusting any number.
20. **New (2026-08-12)**: mel-spectrogram classifier baseline — a small supervised classifier
    trained directly on mel-spectrogram features (not a rendered image, not routed through an
    LALM) for each battery task, as a new baseline sitting alongside the existing L1 DSP floor
    (next action 4) and the L2 encoder probes. Where L1 uses hand-picked DSP estimators
    (autocorrelation, Krumhansl correlation, FFT peak-picking) and L2 probes a pretrained
    encoder's representation, this baseline asks "how much is recoverable from the raw
    time-frequency representation with no hand-designed features and no pretrained encoder at
    all" — a third, complementary floor. Unlike L1, this needs no essentia/beat-tracking
    library, so it's the first floor to cover ALL 13 tasks, including the 3 L1 can't
    (`beats_per_bar`, `progression_id`, `instrument_id`).
    **Built + RUN for real 2026-08-12** (laptop, CPU-only, `musicprobe/mel_baseline.py`):
    log-mel (64 bins), mean+std time-pooled to a fixed 128-dim vector, `LogisticRegression`,
    held out via the same 3-tier split as every LoRA track (`_held_out_mask`, duplicated from
    `image_track_common` to stay torch-free — a first version used a naive random-split
    fallback when no soundfont factor existed, which silently gave `beats_per_bar` a
    non-generalizing split; caught and fixed before trusting the number, same discipline as
    next action 15a). Results, `results/mel_baseline.parquet`:
    | task | acc | chance | split | note |
    |---|---|---|---|---|
    | octave_id | 0.667 | 0.333 | soundfont | |
    | tuning_judgment | 0.741 | 0.500 | base_midi | |
    | instrument_id | 0.667 | 0.250 | soundfont | |
    | beats_per_bar | **1.000** | 0.200 | bpm (genuine held-out-tempo split) | suspiciously
    perfect even under a real tempo-generalization split — checked for a confound directly:
    mean clip duration rises monotonically with beats-per-bar (3-beat ≈11.0s → 7-beat ≈14.6s,
    n=20/class), so this may be reading duration/bar-count structure rather than genuine
    meter perception. Flag as a **task-validity question, not a capability finding**, same
    treatment as the existing beats_per_bar wrong-audio-control oddity (Known gaps) — needs a
    duration-controlled follow-up before citing. |
    | progression_id | 0.417 | 0.250 | soundfont | n=32, hypothesis-generating only |
    | cents_discrimination | 0.389 | 0.333 | base_midi | |
    | note_count | 0.286 | 0.200 | soundfont | |
    | pitch_note_id | 0.367 | 0.083 | soundfont | |
    | chord_quality | 0.167 | 0.125 | soundfont | |
    | interval_id | 0.113 | 0.083 | soundfont | |
    | key_id | 0.075 | 0.042 | soundfont | |
    | mode_id | 0.028 | 0.077 | soundfont | below chance |
    | tempo_bpm | **0.000** | 0.017 | bpm | below chance, and contradicts L1's 0.82 on the
    same audio — **feature-design artifact, not a real finding**: mean/std time-pooling
    destroys periodicity by construction (averaging washes out the very temporal pattern tempo
    lives in), so this baseline is structurally blind to tempo, not evidence tempo is
    unrecoverable from mel features. A time-aware feature (e.g. an onset-strength
    autocorrelation, closer to what L1's own `tempo_estimate` already does) would be needed to
    make this comparison fairly. |
    Reading: consistent with L1/L2 on the near-floor tasks (`mode_id`, `key_id`, `interval_id`
    all weak here too) — three independent "no pretrained encoder / no LALM" methods now agree
    those are genuinely hard, not a probing-method artifact. `beats_per_bar` and `tempo_bpm`
    both need a second look before either number is cited (opposite problems: one may be a
    confound inflating it, the other is a known feature-design blind spot deflating it).
21. **New (2026-08-12)**: run additional music-specific LALMs — MU-LLaMA
    (`github.com/shansongliu/MU-LLaMA`), MusiLingo, and M2UGen (now folded into
    `github.com/shansongliu/MuMu-LLaMA`) — all confirmed to have open weights (checked
    2026-08-12; LLark does not and stays out of scope, same status as GPT-4o-audio). **Caveat
    before committing real setup time**: all three are adapter+LLM heads on top of **frozen
    MERT embeddings** — the same encoder already in this project's Track B roster. Running
    them doesn't open a new encoder axis, it tests whether a different decoder/adapter/LLM
    extracts more from MERT features than the current probes do, which makes this
    conceptually the same question as next action 19 (nonlinear decoder), just answered by
    three full 7B-scale LLMs instead of an MLP head — much more setup cost (three separate
    repos/envs on the H100 box) for a question next action 19 answers more cheaply first.
    Sequence after 19, not before — if a small nonlinear decoder already closes the gap on
    MERT's near-floor tasks, these three add less than they'd otherwise appear to.
    **Harness scaffold built 2026-08-12**: `gpu/eval_music_lalms.py` reuses the exact
    `responses__<tag>.parquet` schema every other Track A model already writes to (same
    `musicprobe.jobs.JOBS_PATH` frozen battery, same resumable-run pattern as
    `musicprobe/runners/run_local.py`) — so once wired up, these three land in the same
    scoring/analysis pipeline with no separate code path. **The three per-model loaders are
    intentionally left as `NotImplementedError` stubs**, not guessed-at inference code — unlike
    Qwen2.5-Omni (already integrated, loading code this project owns and has run), these are
    three third-party repos never loaded here before, and fabricating plausible-looking load
    calls without the actual repos in hand risks silently-wrong inference that looks done but
    isn't (same trap `eval_muchomusic.py`'s BLOCKER section already flags for its own external
    dependency). Whoever runs this on the H100 box: fill in one `_load_*` function per model
    from that repo's own example/demo script — everything else (job iteration, response
    writing, resumability) is ready. Repo URLs and the MERT-overlap caveat are in the file's
    docstring.
22. **New (2026-08-12)**: mechanistic follow-up on why Tracks L-Q/R-W's images actively HURT
    `key_id`/`tempo_bpm` rather than just failing to help — the existing attention diagnostic
    (`gpu/attention_audio.py`, `gpu/plot_attention.py`, already verified eager-attention-safe
    2026-07-25) has never been pointed at these tracks' checkpoints. Question: does attention
    to image tokens spike in a way that correlates with the accuracy drop (a genuine
    "distraction" mechanism, image pulling weight away from audio) or does it stay low like
    the wrong_image≈no_image mechanism control already suggests (model mostly ignores the
    image, and the harm comes from somewhere else — e.g. the modality-dropout training itself
    perturbing the audio-only pathway)? Turns "several representations hurt performance" from
    a bare finding into a mechanistic explanation.
    **Extended 2026-08-12**: `gpu/attention_audio.py` now takes `--lora-checkpoint PATH`
    (+ optional `--tag`) to wrap the qwen-omni preparer's `.thinker` with a saved PEFT adapter
    before running the diagnostic — same wrapping pattern `image_track_common.load_for_eval`
    already uses, additive-only (existing base-model behavior is unchanged when the flag is
    omitted; the preparer functions themselves weren't touched). Syntax-checked, not run — no
    GPU here to test against a real checkpoint.
    **Dependency to verify first, still open**: whether Track L-Q/R-W's per-seed checkpoints
    (`gpu/train_track_repr.py`'s `ckpt_subdir`) are still on the H100 box or were cleared after
    eval — if cleared, this needs a rerun of at least one seed before the diagnostic can attach.
    Whoever has H100 access should check this before scoping further; not resolvable from the
    laptop.
23. **New (2026-08-12)**: does the D-zoom/E fix (which worked on clean synthetic tones) hold up
    on REAL recordings — real instruments/vocals/mixes, not MIDI+soundfont? Every task in this
    project so far uses synthesized stimuli specifically because they give free perfect ground
    truth (RESEARCH_PLAN.md Sec3, Tier 1). This asks the generalization question directly:
    same front-end (Track D-zoom's zoomed+referenced image, Track E's pitch-as-text), same
    trained checkpoint, fed real audio it never saw a rendering of during training.
    **Dataset hunt (2026-08-12) — 3 of 4 candidates dead/gated, tested directly, not assumed**:
    - GiantSteps Key dataset (real EDM tracks, expert key labels): its documented Beatport CDN
      download mechanism was tested against 5 different track IDs — 404 on all of them, audio
      hosting has rotted since 2015. Its HuggingFace re-upload (`m-a-p/GS`) was checked too —
      mirrors only the same annotations + the same broken script, no actual audio either.
      **No real-music key_id source currently available** — this closes off next action 21's
      "both" scope from the earlier discussion down to pitch/interval only, for now.
    - MAESTRO (real piano performances + aligned MIDI): metadata catalog (1276 tracks) is live,
      but individual audio file URLs tested 404 — audio may only ship inside a ~100GB bundle,
      not practical here.
    - NSynth (real individually-recorded instrument notes): confirmed LIVE via HuggingFace
      `datasets` (`confit/nsynth-parquet`) — reached the actual audio-decode step before
      hitting a missing (fixable) library. Real audio, but isolated 4-second notes, not full
      pieces — set aside since it doesn't test the actual generalization question (Rupali's
      call: prioritize real full-piece audio over this).
    - **MedleyDB (real multitrack songs, expert melody F0 annotation)**: audio requires a
      manual Zenodo access request — **Rupali is requesting access directly**; this next
      action builds everything else now so it's instant to run once access lands.
    **Built 2026-08-12**: `musicprobe/real_music_medleydb.py` — segments MedleyDB's continuous
    melody F0 curve into sustained-note events (groups frames within 35 cents of a running
    median, unlike a plain voiced/unvoiced split, since a real melody's pitch wanders
    continuously), builds `pitch_note_id` stimuli from single segments and `interval_id`
    stimuli from adjacent segment pairs (same `INTERVALS` vocabulary as the synthetic battery,
    so ground truth is directly comparable), reuses `build_prompt` for identical prompt
    phrasing. **No key ground truth** — MedleyDB's own metadata has no key label, and this
    module deliberately does not guess one from track titles (that would be exactly the kind
    of unverified ground truth the L1/L2/L3 discipline exists to avoid).
    **Contamination control (Rupali's call)**: reuses this project's existing `wrong_audio`
    mechanism rather than a new design — every real-music job is paired with the identical
    question against a mismatched real clip, scored against the original ground truth. If a
    model answers a famous-song question right on the wrong clip, that's text-prior recall,
    not listening — same logic already applied to every synthetic task.
    **Verified WITHOUT real MedleyDB data** (none exists on this laptop yet), two ways:
    (1) `--selftest`: synthetic 3-note melody (C4→E4→G4, major third then minor third, with
    vibrato) — confirms segmentation recovers exactly 3 segments and the correct interval
    labels, and that Track D-zoom's `render_zoom`/Track E's `f0_text` run unmodified against
    non-battery audio. (2) a fake `mirdata`-shaped mock dataset (2 fake tracks) exercised
    `build_manifest()` itself end to end — correct per-track segmentation, snippet audio
    actually written to disk, correct interval labels (major third / minor third), and a
    48-job table with a clean 24/24 audio/wrong_audio split. Both pass. **Still needed once
    real data lands**: run `build_manifest(data_home=...)` for real, render Track D-zoom/E's
    front-ends against the resulting snippets (both functions are already generic, no changes
    needed), and run the ALREADY-TRAINED Qwen2.5-Omni D-zoom/E checkpoints against them in
    eval-only mode — this is inference on an existing checkpoint, not a new training run, but
    still needs a small eval-only entry point pointed at
    `manifests/real_music_medleydb_jobs.parquet` instead of the synthetic jobs table (not
    built yet — low priority until the data itself is in hand).
    **Interim real-recordings set, built + tested 2026-08-12 (while MedleyDB access is
    pending)**: 7 real classical recordings pulled from Wikimedia Commons (free-licensed,
    no login/API gate — unlike GiantSteps/MedleyDB), a mix by design: 4 "famous" pieces
    where key is well-documented but a model could plausibly recall it from text alone
    (Beethoven Moonlight Sonata 1st mvt, Beethoven Für Elise, Pachelbel Canon in D, Chopin
    Nocturne Op.9 No.2) and 3 "obscure" pieces with equally solid documented ground truth but
    far lower text-memorization risk (2 Scarlatti keyboard sonatas identified only by
    Kirkpatrick catalog number, a Clementi pedagogical sonatina). Stored at
    `stimuli/real_recordings/` + `manifests/real_recordings_manifest.{csv,parquet}` (title,
    composer, key, key source/citation, fame tier, contamination-risk note, source URL,
    license — every field traceable, no guessed ground truth). All 7 verified playable
    (one, the Chopin file, needed re-encoding from a skeleton-multiplexed Ogg Vorbis stream
    libsndfile couldn't parse to a plain wav via librosa's audioread fallback — noted in case
    it recurs with other Commons files).
    **Ran the same L1-key-estimate + note-segmentation pipeline test as the Bach/Debussy pass
    (2026-08-12) against all 7**: key_estimate matched the documented key on 4/7 (Moonlight
    Sonata, Für Elise, Canon in D, Scarlatti K.87) and missed on 3/7 — Chopin Nocturne
    (documented E♭ major, guessed G minor), Scarlatti K.466 (documented F minor, guessed C
    major), Clementi Sonatina (documented C major, guessed G major — the dominant, a classic
    naive-key-detector confusion, most explainable of the three misses). No fame-tier pattern
    in the misses (2 famous, 1 obscure) — this is L1's *own* algorithm generalizing
    imperfectly to real (non-monophonic-synthetic) audio, not yet a statement about the
    trained LALM, which still hasn't been run against any of this (no GPU here). All 7 produce
    7–37 clean monophonic note-segments in their first 30s via the same segmenter used for
    MedleyDB, so note/interval snippets (same recipe as the Bach bass-line test) can be pulled
    from any of them on request.
24. **New (2026-08-12)**: does the D-zoom/E fix specifically (the front-end that took
    `cents_discrimination` 0.55→0.94 and `tuning_judgment` 0.53→0.89 on synthetic tones,
    PAPER.md) hold up when the base tone is REAL recorded timbre? Ground-truth reality check
    first: "is this note 37 cents flat" has no naturally-occurring answer outside a controlled
    synthesis — that's *why* the original battery generated these two tasks from scratch
    rather than using recordings for anything. Design (Rupali's call, stated plainly in every
    stimulus's own `provenance` field, not hidden): real recorded note as the base timbre +
    the SAME kind of exact controlled digital pitch-shift the synthetic stimuli always
    needed. Real timbre, still-exact ground truth — this tests the front-end's generalization
    to real timbre specifically, not a claim that the whole stimulus is unmodified real audio.
    **Source**: NSynth ('pitch' config, `confit/nsynth-parquet` on HuggingFace, confirmed live
    2026-08-12 per next action 23's dataset hunt) — pulled via `datasets.Audio(decode=False)`
    + manual `soundfile` decode, deliberately avoiding `torchcodec`/`torch` so this laptop
    stays GPU-stack-free like every other CPU-side module here.
    **Built + run for real 2026-08-12**: `musicprobe/real_music_nsynth.py` — 180 stimuli (60
    each `pitch_note_id`/`cents_discrimination`/`tuning_judgment`) across real acoustic,
    electronic, and synthetic-synth instrument families (organ, keyboard, bass, brass, guitar,
    etc.), 1080 jobs with the same `audio`/`wrong_audio` split as every other track. Applied
    the SAME psychometric ladder as the original battery (5/10/25/50/100-cent deltas,
    TASKS.md 1.8) and the same 25-cent in-tune/out-of-tune threshold as
    `l1_baselines.tuning_estimate`, called unmodified.
    **L1 comparison (same estimators, same code, run against this hybrid audio instead of
    pure synthetic)**:
    | task | L1 acc on real+shifted | L1 acc on original synthetic battery | |
    |---|---|---|---|
    | `pitch_note_id` | 0.733 (n=60) | 1.00 | degrades on real timbre — autocorrelation is
    noisier against real harmonic/inharmonic overtone structure than a clean synthesized tone |
    | `cents_discrimination` | 0.800 (n=60) | 1.00 | also degrades, smaller drop |
    | `tuning_judgment` | 0.633 (n=60) | 0.63 (next action 4's table) | **no real-vs-synthetic
    gap at all** — matches the original doc's own read of this number ("noisy, naive
    threshold, not tuned"); this task's L1 estimator is equally imprecise on both, so this
    specific task's difficulty isn't a real-audio-generalization story |
    One real bug caught during this run, not silently patched around: `l1_baselines.f0_autocorr`
    can return a non-finite/negative value on some real (non-clean-synthetic) inputs at its
    lag-boundary edge cases — never triggered by the synthetic-only battery, so never guarded
    against. Fixed with a local validity check in `real_music_nsynth.l1_accuracy`, NOT by
    editing `l1_baselines.py` itself (same "don't change already-verified code under
    already-reported numbers" discipline as every prior track).
    **Front-end confirmed rendering correctly on this hybrid audio** (not just L1): Track
    D-zoom's `render_zoom` and Track E's `f0_text` both ran unmodified against the real+shifted
    clips — `f0_text` on a `cents_discrimination` "lower" example read `232.4, 228.4` Hz for
    tone1/tone2, correctly reflecting the intended downward shift.
    **Still needed**: the actual GPU eval-only pass with the trained D-zoom/E checkpoints
    against `manifests/real_nsynth_jobs.parquet` — same status as next action 23, needs the
    H100 box, not built yet (low priority to build blind before the data's usefulness is
    confirmed by a first real run).
25. **New (2026-08-12, Rupali relaying the professor's framing)**: for any near-floor task
    (`mode_id`, `key_id`, `interval_id`, next action 13), is the information (a) genuinely
    absent from the encoder at every depth, or (b) present in early/mid layers but discarded
    by the final layers the existing probes all read from — the "late-layer loss" question,
    where the fix would be reading an earlier layer instead of the last one? Next action 19
    (nonlinear decoder) adds a third possibility this framing leaves out: (c) present at some
    layer but linearly inseparable, fixable with a smarter reader, not a different layer.
    Explicitly asked to check this **per task, on real music**, not assume one answer covers
    every task ("maybe it varies test to test").
    **Built 2026-08-12**:
    - `gpu/probe.py` and `gpu/probe_mlp.py` both gained a `--group-key` flag (default
      `soundfont`, unchanged — every already-run/verified probe call keeps its exact original
      behavior). Real-music manifests have no soundfont; `real_music_medleydb.py` and
      `real_music_nsynth.py` now both emit a `factors` column with the right substitute —
      `track_id` for MedleyDB (never split one real song's notes across train/held), 
      `instrument_family` for NSynth (parsed from NSynth's own filename convention, e.g.
      `organ_electronic`) — same leakage-guard role as soundfont, re-verified against both the
      MedleyDB selftest/fake-mock and a live NSynth regeneration after the change; nothing
      broke.
    - `gpu/classify_layer_pattern.py` — takes a linear (`probe.py`) and nonlinear
      (`probe_mlp.py`) per-layer CSV for the same task/encoder/manifest and returns one of
      five verdicts: `NEVER_CAPTURED` (neither clears chance anywhere — genuine absence),
      `LATE_LAYER_LOSS` (linear clears chance early/mid, drops to chance in the final layers —
      the professor's option (a), read off an earlier layer), `PRESENT_THROUGHOUT` (no late
      decline — representation isn't the bottleneck), `NONLINEAR_ONLY` (linear never clears
      chance, nonlinear does — present but linearly inseparable, next action 19's case), or
      `MIXED` (doesn't fit cleanly, reported as such rather than forced into a bucket). "Clears
      chance" = accuracy ≥ chance + 0.08, an absolute margin chosen to reproduce this project's
      own existing judgment call on a real number (mode_id's 0.04–0.12 vs chance 0.077 stays
      below this margin, matching the "barely above chance" language already used for it in
      next action 13 — not a threshold invented from nothing). **Verified against 5
      constructed synthetic curves, one per verdict shape — all 5 classify correctly**
      (`--selftest`).
    - `extract_activations.py` needed **no changes** — it already takes `--manifest`, so it's
      already compatible with `real_music_medleydb.parquet`/`real_nsynth_manifest.parquet` as
      soon as either has real audio behind it (checked directly, not assumed).
    **Still blocked**: this whole diagnostic needs real per-layer activations from the actual
    trained model's own audio encoder on real audio — `extract_activations.py --own-encoder`
    against MedleyDB (once Zenodo access lands) or NSynth (ready now), then `probe.py`/
    `probe_mlp.py --group-key ...` per task, then `classify_layer_pattern.py` per result pair.
    Every piece up to that point is built and verified; the GPU run itself is the one
    remaining step, same H100-box blocker as next actions 17/19/21/22/23/24.

## Known gaps / honesty list
- **Methodological stance on wrong_image/wrong_audio/image_wrong_audio, decided 2026-08-12
  (Rupali's call)**: keep collecting these conditions in every future track — cheap, already
  built into the job schema, no reason to stop — but stop treating "wrong_X ≈ no_X" alone as
  proof the model "ignores" or "doesn't read" that modality. The test can't distinguish never-
  processed from processed-then-discounted (a model could recognize a mismatched image and
  learn to disregard it, producing the identical wrong_image≈no_image signature as never
  looking at all) — both explanations are behaviorally indistinguishable from this control by
  itself. Going forward: (1) prefer "the image/audio doesn't measurably change accuracy" over
  "the model ignores/doesn't read it" in write-ups — the first is what the test actually
  supports, the second overclaims a specific mechanism; (2) treat wrong_X results as one
  signal to corroborate with others (L2 probes, the nonlinear-decoder/layer-pattern
  diagnostic from next action 25, replication across seeds/tasks), not the sole basis for a
  mechanism claim. Retroactive note, not a retraction: existing findings that lean on this
  (Track D-force's "forces image use," D-zoom/E's "substitution not hearing," Track H's
  "not comparing in audio") were reasonable calls under the discipline used at the time and
  aren't being walked back — this changes the bar for *new* claims from here on.
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

# Tonight's session results — 2026-08-21, RunPod A100 80GB

Branch: `rupali-music`. Machine: RunPod pod, driver 570.172.08, CUDA 12.8,
torch 2.6.0+cu124. Model: Qwen2.5-Omni-7B, base (no fine-tuning) unless noted.

## Setup

- Stimulus generation + self-test: **PASSED**, 0 errors across 2,208 jobs.
  Manifest content-verified identical to the frozen version (git binary diff
  showed a row-order/metadata difference only, not a real data discrepancy).
- 7 real, environment-level bugs found and fixed (all committed to
  `rupali-music`): CUDA/torch version mismatch (cu130 default vs. this pod's
  12.8 driver), missing torchvision, torchcodec version incompatibility
  (worked around via `Audio(decode=False)` + manual soundfile decode instead
  of chasing versions), a `disable_talker()` crash affecting 5 files
  (including the shared loader Track Z/D-zoom both import), and a missing
  f0-zoom image render prerequisite for D-zoom retraining.

## PitchBench (external, published benchmark — real audio, isolated pitches)

- 5-item smoke test: 0/5 correct. Raw responses checked — clean parsing, not
  a bug. Model repeated the same wrong answer (48) for 3/5 genuinely
  different ground-truth pitches.
- 100-item capped run (`pitchbench_a1_single_pitch_id`): **19/100 = 0.190**.
  Parsing verified clean (0 unparseable responses out of 100). Consistent
  with PitchBench's own published finding that absolute pitch hearing is
  weak even in larger/newer models.

## Track Z (self-transcription auxiliary objective)

- Smoke test only (8 steps) — **paused per your call**: doesn't test the
  evidence-grounding contribution directly, deprioritized in favor of the
  Suite C work. Loss finite throughout (1.108 → 0.8828), trains cleanly, no
  crash. No real (non-smoke) run attempted.

## Track D-zoom (zoomed F0-contour + reference line — the strongest prior result)

- Smoke test (8 steps): PASSED, loss finite (0.1945 → 0.01937).
- **Full training run (3 epochs, 132 steps, 7.2 min, ~$0.17)**: succeeded.
  Checkpoint saved (`track_d_force_ckpt/qwen25omni-zoom-s0/`).
- **Held-out synthetic battery, seed 0 ONLY (isolated from 2 pre-existing
  committed seeds to keep this an honest fresh-reproduction check)**:

  | Task | Audio-only | Audio+image |
  |---|---|---|
  | cents_discrimination | 0.538 | 0.923 |
  | tuning_judgment | 0.556 | 0.926 |
  | octave_id | 0.800 | 0.833 |
  | note_count | 0.893 | 0.857 |

  This independently reproduces the original 2026-07-29 result (cents
  0.55→0.94, tuning 0.53→0.89) from a completely fresh training run on this
  new pod — a real, credible confirmation, not a fluke.
- **Not yet done**: the real-NSynth generalization test (does this fix
  survive real instrument timbre, not just synthetic tones) — harness is
  built (`eval_track_dzoom_real.py`), checkpoint now exists, just hasn't been
  run yet.

## Real-recordings key-ID eval (genuine real music — full performances, not isolated notes)

7 free-licensed Wikimedia Commons pieces, base model, asked "what key is this
piece in?", scored against documented/cited keys (not a musician's judgment).

| Piece | Fame | Documented key | Model's answer | Correct? |
|---|---|---|---|---|
| Moonlight Sonata | famous | C# minor | C# minor | ✓ |
| Für Elise | famous | A minor | A minor | ✓ |
| Canon in D | famous | D major | D major | ✓ |
| Chopin Nocturne Op.9/2 | famous | E♭ major | C minor | ✗ |
| Scarlatti K.466 | obscure | F minor | F minor | ✓ |
| Scarlatti K.87 | obscure | B minor | B minor | ✓ |
| Clementi Sonatina | obscure | C major | C major | ✓ |

**6/7 correct (n=7, too small to generalize the number) — but the reasoning
given is the real finding**: on 3 separate items, the model justified its
(correct) answer with a "key signature" claim that is *factually wrong on its
own terms* (e.g., claimed C# minor has "seven sharps" — actually 4). Key
signatures aren't audible in the first place, so this is confabulated
reasoning, not genuine audio-grounded justification — a concrete, citable
example of "sounds knowledgeable without being audio-grounded."

## Grounding pilot (Suite C) — planning only, nothing built

`GROUNDING_PILOT_PLAN.md` — evidence schema, 4 phenomena, natural-language
question phrasing, 3-tool decision (pitch/rhythm/harmony), sharpened success
criterion (accuracy improvement + swap-sensitivity, not either alone). Fully
reviewed and revised; ready to move into building.

## Also confirmed existing (not built tonight)

- **Suite A (retrieved knowledge)**: `benchmark/tier1_pilot.json` /
  `tier2_pilot.json` — 5 songs, ~27 citation-gated facts, already rigorous,
  built 2026-08-16.

## Cost

Pod up ~8h45m tonight (rough estimate from the pod's own clock, not an
authoritative billing pull — check the RunPod dashboard for the real number).
Actual GPU-active time across all runs above: well under 30 minutes total.

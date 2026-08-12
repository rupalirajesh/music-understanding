# Music Understanding in Audio LLMs — Background & Research Plan

A deep-dive study of how audio-language models represent, process, and reason about music —
across all genres — with the goal of building the knowledge base to train a music
understanding model.

Last updated: 2026-08-05 (Tracks G/H results — chromagram + in-audio reference tone, both null; H7 resolved, §12)

---

# PART 0 — CONCEPTS FROM SCRATCH (onboarding primer)

This part assumes no machine-learning background. It builds every concept the rest of the
plan uses, in order, and ends with a walkthrough of what actually happens when we "run an
experiment." Read this before Part I; Part I then adds the audio-specific detail.

## 0.1 What a language model is

A **large language model (LLM)** is a program trained to do one thing: given a sequence of
text, predict what comes next. It was trained by showing it a huge fraction of the internet
and adjusting its internal numbers (**parameters** or **weights** — billions of them) until
its predictions got good. Everything an LLM appears to "know" is stored implicitly in those
weights.

Some vocabulary that falls out of this:

- **Token**: models don't read letters or words; text is chopped into pieces called tokens
  (roughly ¾ of a word each). "Understanding music" might become three tokens. The model's
  input and output are sequences of tokens.
- **Embedding**: each token is converted into a long list of numbers (a **vector**, e.g.
  4,096 numbers). This vector is the model's internal "meaning" of that token. Similar
  meanings end up as nearby vectors — this is the key trick that makes everything else work.
- **Layers and hidden states**: the model is a stack of identical processing blocks
  (**layers** — typically 30–80 of them). Each layer takes in the vectors from the previous
  layer, lets every position look at every other position (**attention**), and outputs
  refined vectors. The vectors at each intermediate layer are called **hidden states** or
  **activations**. Early layers hold surface-level features; deeper layers hold more
  abstract ones. *We will be reading these hidden states directly — this is what "probing"
  means (§0.5).*
- **Inference**: running a trained model to get answers (as opposed to training it).
- **Temperature**: a knob controlling output randomness. Temperature 0 = always pick the
  most likely next token = deterministic. We run all evals at temperature 0 so results are
  reproducible.
- **Prompt**: the input text we give the model, including instructions and the question.

**How LLMs become assistants:** raw next-token prediction produces a model that continues
text, not one that answers questions. So after pretraining, models get **instruction tuning
/ SFT (supervised fine-tuning)**: further training on examples of (question → good answer)
pairs. This matters enormously for us, because *what's in those pairs determines what the
model can talk about* — a recurring theme in this study (§1.4).

**Open-weights vs closed (API) models:** for some models (Qwen, Audio Flamingo) the weights
are downloadable — we can run them on our own GPU, read their hidden states, and fine-tune
them ("**white-box**"). Others (Gemini, GPT-4o) are only reachable over the internet via an
**API**: we send a prompt + audio, get text back, and see nothing inside ("**black-box**").
Track A works on both; Tracks B and C need open weights.

## 0.2 How audio exists in a computer

- **Waveform / samples**: a microphone measures air pressure thousands of times per second.
  Each measurement is a number; the list of numbers is the waveform. The **sampling rate**
  is how many measurements per second — CD/streaming music uses 44,100 per second (44.1 kHz).
- **Nyquist limit**: a recording sampled at rate R can only represent frequencies up to R/2.
  So 16 kHz audio contains nothing above 8 kHz. Most audio models resample everything to
  16 kHz, silently discarding the top of the spectrum (cymbal shimmer, harmonics, "air").
- **Spectrogram**: instead of pressure-over-time, break the audio into tiny windows
  (~25 ms) and compute which frequencies are present in each — a picture with time on the
  x-axis and frequency on the y-axis. A **mel spectrogram** spaces the frequency axis the
  way human hearing does (finer at low frequencies). This image-like representation is what
  most audio models actually consume, not the raw waveform.
- **Cents**: the unit for fine pitch differences. 100 cents = 1 semitone (adjacent piano
  keys). Trained musicians hear ~5–10 cent differences. When we measure a model's pitch
  discrimination "threshold in cents," we're asking how fine its pitch hearing is.
- **12-TET**: standard Western tuning divides the octave into 12 equal semitones. Much
  real-world music (blues bends, vocal slides, Indian classical) lives *between* these
  steps — one of our probes asks whether models' internal pitch representation is snapped
  to the 12-TET grid.

## 0.3 How a model "hears": the audio LLM (LALM) recipe

An LLM only understands token vectors. To let it hear, you bolt on two components:

```
audio → [encoder] → sequence of vectors → [bridge/adapter] → vectors the LLM can read
                                                                     ↓
                                    LLM reads them like a prefix of "audio words",
                                    then answers your text question about them
```

1. **Audio encoder**: a separate neural network, trained beforehand on audio-only tasks,
   that converts a clip into a sequence of vectors — one vector per small time slice. Think
   of it as the model's ear. Crucially, *the encoder can only pass along what its own
   training taught it to notice*. Most encoders were trained for speech recognition
   (transcribing words), where knowing a note is C vs C# was never necessary. (§1.2 covers
   the encoder families.)
2. **Bridge / adapter / projector**: a small translation layer that maps the encoder's
   vectors into the LLM's embedding space, so the LLM can treat them like tokens.
3. **Token rate**: the encoder emits vectors at some fixed rate, e.g. 12.5 per second =
   one vector per 80 ms. Anything that happens *faster* than that (a quick ornament, a
   grace note) must be summarized into a single vector or it's simply gone. This is a hard
   information bottleneck and one of our central suspects (§1.1, §1.5).

The whole assembly — encoder + bridge + LLM — is called an **audio-language model**
(**LALM** = Large Audio Language Model). "Omni" models are the same idea with audio tokens
integrated more natively.

**The pipeline is a chain of custody for musical information.** The note is in the air →
does it survive resampling to 16 kHz? → does the encoder represent it? → does the bridge
pass it through? → does the LLM *use* it when answering? Our whole methodology (§0.5) is
about finding the link where the chain breaks.

## 0.4 Fine-tuning, LoRA, and why we'll do surgery on models

- **Fine-tuning**: taking a trained model and training it a bit more on your own examples
  to teach it a specific skill. Full fine-tuning updates all billions of weights —
  expensive.
- **LoRA (Low-Rank Adaptation)**: a cheap fine-tuning trick. Freeze the original weights;
  attach small trainable "patch" matrices alongside them and train only those (<1% of the
  parameters). Runs on a single Colab GPU. When the plan says "three-arm LoRA fine-tune"
  (§6), it means: three separate LoRA training runs that patch *different parts* of the
  model (just the LLM side; LLM + encoder; and a control), to see which patch fixes a
  failure — which tells us where the failure lived.
- **Frozen**: a component whose weights are not updated during a given training stage.

## 0.5 The measurement toolkit — the ideas behind L1 / L2 / L3

This is the intellectual core of the study. For any musical property (say, "what key is
this in?"), we ask three separate questions:

**L1 — Is the information physically in the audio?**
Use classical signal processing / small purpose-built tools (**MIR** = Music Information
Retrieval, the pre-LLM field of extracting musical facts from audio: pitch trackers, beat
trackers, key detectors). If a standard key-detection algorithm gets it right, the key is
recoverable from the signal. L1 is our ground-truth sanity floor.

**L2 — Is the information inside the model's head?**
This is **probing**. Take the model's hidden states (§0.1) for our audio clips at some
layer. Train a *tiny, simple* classifier — a **linear probe**, essentially one-step
logistic regression from scikit-learn — to predict the property (the key, the chord
quality) *from those hidden states alone*. If a classifier that simple can read the key off
the model's internal vectors, the information is demonstrably *encoded in there*. We do
this layer by layer ("layer-wise probing") to see where in the stack information appears,
peaks, or fades. The probe must be simple on purpose: a powerful probe could compute the
answer itself, which would prove nothing about the model.

**L3 — Does the model behaviorally use it?**
Just ask the model the question in plain language and score its answer. This is standard
**behavioral evaluation** — no internals needed, works on API models too.

**Why three levels?** Because the *pattern* across them is diagnostic:

- **L1 ✓, L2 ✓, L3 ✗** — the model's ear hears it, but the language side can't talk about
  it. We call this an **alignment gap**. Prediction: cheaply fixable with a little
  fine-tuning (Track C tests exactly this).
- **L1 ✓, L2 ✗** — the signal contains it but the encoder never learned to represent it.
  The problem is the encoder's training or the architecture (token rate, 16 kHz). No amount
  of instruction data fixes this.
- **L2 ✗, L3 ✓** — red flag! The model answers correctly *without* the information being
  in its audio representation — meaning it's guessing from **text priors**: statistical
  knowledge of what answers are plausible ("a question about a sad slow song is probably in
  a minor key"), not listening. The MuChoMusic benchmark famously caught models scoring
  well on music questions *with the audio removed*. This is why every task carries a
  **no-audio control**: run the identical question without the audio; if accuracy barely
  drops, the task was never measuring hearing.

The three experimental tracks map onto this: **Track A** = L3 at scale (behavioral,
black-box, all models). **Track B** = L2 (probing, open models). **Track C** = causal
follow-up — if probing says "the info is in there," prove it by fine-tuning the alignment
and watching behavior recover.

## 0.6 Experimental hygiene concepts (the controls vocabulary)

These terms appear throughout §4 and §9; here's what they mean and why they exist.

- **Synthetic stimuli**: audio we generate ourselves from **MIDI** (a symbolic score format
  — a list of note-on/note-off events, no sound) rendered to audio with a synthesizer
  (fluidsynth + **soundfonts** = libraries of instrument sounds). Because we wrote the
  score, ground truth is *perfect* — we know the exact key, tempo, and every note. It also
  gives **factor control**: we can change one variable (transpose the key) while holding
  everything else constant.
- **MCQ vs open-ended**: multiple-choice questions are easy to score automatically but
  models exploit their structure (position biases, elimination); open-ended answers are
  more natural but need scoring by a text LLM acting as **judge** — which we validate by
  hand-scoring ~100 items and checking the judge agrees.
- **Label/position permutation**: shuffle answer-choice order across items so "always pick
  B" can't score above chance.
- **Paraphrases**: ask each question ≥3 different ways, so we measure the skill, not
  sensitivity to one phrasing.
- **Confusion matrix**: a table of which wrong answers occur. Errors carry signal: calling
  G major "D major" (adjacent on the circle of fifths) is an honest listening error;
  uniformly random errors mean guessing. Same logic for tempo (double/half-tempo "octave
  errors") and technique (bend↔slide vs bend↔palm-mute).
- **Psychometric curve**: borrowed from human hearing science. Vary difficulty continuously
  (pitch difference of 5, 10, 25, 50, 100 cents) and plot accuracy vs difficulty. The point
  where the curve crosses threshold is the model's discrimination limit — far more
  informative than a single accuracy number.
- **Training-data contamination**: famous datasets (GTZAN, MusicCaps) were probably *in*
  these models' training data, so good scores may be memorization. Defense: synthetic
  stimuli and recordings released after the model's training cutoff.
- **Leakage** (in probes): if clips from the same singer appear in both the probe's train
  and test sets, the probe can cheat by recognizing the voice. So we split by singer /
  soundfont / session, never just by clip.
- **Held-out / transfer set**: data never seen during (fine-)tuning, used to check the
  model learned the skill rather than the examples — including transfer from synthetic
  training stimuli to real recordings.

## 0.7 What running an experiment actually looks like

Concretely, the day-to-day loop. Everything is driven by one **manifest** file
(`stimuli.parquet` — a big table): each row = one stimulus, with its audio path, task name,
ground-truth answer, factor values (key, instrument, tempo…), and which controls apply.

**Step 1 — Make stimuli (local machine).** Python scripts generate MIDI, render it through
fluidsynth with several soundfonts, apply manipulations (transposition, detuning), compute
ground truth, and append rows to the manifest. For real-music tasks, rows point into
datasets like MTG-Jamendo with their labels.

**Step 2 — Run models (Track A / L3).** The harness walks the manifest: for each row it
builds the prompt (with paraphrase and choice-permutation variants), sends audio + question
to the model — API call for closed models, local inference on a Colab GPU for open ones —
at temperature 0, and logs *everything* (model version, prompt, audio hash, raw answer) to
a results table. No-audio and wrong-audio control rows run through the same pipe.

**Step 3 — Score & analyze (local).** Parse answers (or run the LLM judge for open-ended),
join against ground truth, and produce per-task accuracy, control-corrected accuracy,
confusion matrices, and psychometric curves. Deliverable: the capability heatmap
(task × model).

**Step 4 — Extract activations (Track B / L2, Colab).** Load an open model, play each
stimulus through it, and save the hidden states at every layer to disk (they're just big
arrays). This is the GPU-hungry step; probing itself is cheap.

**Step 5 — Probe (local).** For each (layer × property): train a linear probe on the saved
activations with scikit-learn, evaluate on the leakage-safe split, plot
probe-accuracy-by-layer. Combine with Step 3 into the L2/L3 dissociation table — the
study's centerpiece.

**Step 6 — Intervene (Track C, Colab).** For the clearest L2-high/L3-low failures:
programmatically generate a small instruction dataset, run the three LoRA arms, re-run the
Step 2–3 evaluation on held-out stimuli, and see which arm closes the gap.

The person running experiments mostly lives in Steps 1–3 and 5 (Python, pandas,
scikit-learn — no deep learning expertise needed) with Steps 4 and 6 being
recipe-following on Colab. The harness in Step 2 is the first thing we build (Phase 0),
and validating it by replicating published benchmark numbers is how we know our plumbing
is trustworthy before we trust any novel result.

## 0.8 Mini-glossary (quick reference)

| Term | One-liner |
|---|---|
| LLM | Text-in/text-out neural network trained to predict the next token |
| Token | Chunk of text (~¾ word); the model's unit of input/output |
| Embedding / vector | List of numbers representing a token's meaning |
| Hidden state / activation | The vectors inside the model at some layer |
| Layer | One of the model's stacked processing blocks (30–80 total) |
| Encoder | Audio-in/vectors-out network — the model's "ear" |
| Bridge / adapter | Translator from encoder vectors to LLM embedding space |
| LALM | Audio LLM: encoder + bridge + LLM |
| Token rate | Audio vectors per second reaching the LLM (e.g. 12.5 Hz = 80 ms each) |
| SFT / instruction tuning | Training on (question → answer) pairs to make a model helpful |
| LoRA | Cheap fine-tuning: train small patch matrices, freeze the rest |
| Linear probe | Tiny classifier reading a property from hidden states → "is it encoded?" |
| L1 / L2 / L3 | Info in signal / in representation / in behavior |
| Alignment gap | L2 ✓ but L3 ✗ — model hears it but can't say it |
| Text prior | Answering from plausibility instead of listening |
| No-audio control | Same question, audio removed — detects text priors |
| MIR | Music Information Retrieval — classical algorithms for musical facts |
| MIDI / soundfont | Symbolic score format / instrument-sound library for rendering it |
| Cents | Fine pitch unit; 100 cents = 1 semitone |
| Psychometric curve | Accuracy vs difficulty; threshold = perceptual limit |
| Contamination | Eval data that was in the model's training set |
| Leakage | Train/test overlap letting a probe cheat (same singer, etc.) |
| Temperature 0 | Deterministic model output (reproducible evals) |
| White-box / black-box | Weights accessible (probe, tune) vs API-only (behavior only) |

---

# PART I — BACKGROUND: HOW THESE MODELS WORK

Read this part top-to-bottom once; it's ordered from signal-level basics to the current SOTA.
Section 1.7 is the reading list in recommended order.

## 1.1 Audio representations 101 (the substrate everything sits on)

- **Waveform**: pressure samples over time. Music is distributed at 44.1 kHz; **almost every
  audio LLM resamples to 16 kHz**, which caps representable frequencies at 8 kHz (Nyquist).
  Fundamentals of musical notes live below ~4 kHz, but upper harmonics, cymbal/air content,
  and production detail are discarded before the model ever "hears" the clip. This is the
  first, easily-forgotten bottleneck.
- **Mel spectrogram**: short-time Fourier transform → perceptually-spaced frequency bins.
  Whisper-family front-ends use 128 mel channels, 25 ms window, 10 ms hop. Mel spacing is
  speech-motivated; adjacent semitones in low registers can fall into the same mel bin.
- **CQT (constant-Q transform)**: log-frequency bins aligned with musical pitch — each
  semitone gets equal resolution. Rarely used in LALM front-ends, but used as a *training
  target* in MERT (below) precisely to inject pitch/harmony inductive bias.
- **Frame rate / token rate** — the single most important number for our study. The encoder
  emits a sequence of vectors at some rate; after pooling/downsampling the LLM sees audio
  "tokens" at:
  - Whisper encoder: 50 Hz native, typically pooled 2–4× → 25 or 12.5 Hz at the LLM
  - Qwen3-Omni's AuT encoder: **12.5 Hz** (80 ms per token)
  - MERT: 75 Hz (it's an encoder for MIR, not an LLM front-end)
  - Neural codecs (EnCodec/DAC): 50–75 Hz × multiple codebooks
  
  For scale: a 16th note at 120 BPM lasts 125 ms ≈ **1–2 tokens** at 12.5 Hz. A vibrato
  cycle (~5–7 Hz) spans a handful of tokens; a fast ornament may fall entirely inside one.
  Anything the encoder doesn't summarize into that one vector is gone. This predicts a
  specific failure class (fast/fine pitch events) that our probes should target.

## 1.2 The four audio-encoder families

Every music-capable LLM is built around one of these. Their pretraining objective determines
what musical information survives into the representation.

**(a) Supervised speech encoders (the dominant choice).**
- **Whisper** (large-v3): encoder-decoder trained on ~5M hours of weakly-supervised
  transcription/translation. The encoder is a superb general audio feature extractor *for
  speech*; its objective never required distinguishing C4 from C#4, only phonemes and
  prosody. Yet it is the backbone of Qwen2-Audio, SALMONN, AF3, and most LALMs.
- **AF-Whisper** (Audio Flamingo 3): Whisper-large-v3 *further pretrained* on sound + music
  + speech (captioning-style objectives) to become a unified encoder; NVIDIA showed one
  unified encoder beats separate speech/sound dual encoders at equal data budget.
- **AuT** (Qwen3-Omni): attention encoder-decoder ASR model trained *from scratch* on
  20M hours of supervised audio; 8× downsampled 128-dim fbank → 12.5 Hz token rate.

*Implication for us:* the field's default encoders are ASR-shaped. Precise-pitch failures
may originate here, not in the LLM.

**(b) Self-supervised (SSL) encoders.**
- **wav2vec 2.0 / HuBERT / BEATs**: masked prediction of quantized/clustered units. General
  audio, speech-leaning.
- **MERT** (the music-specific one, and our main probing substrate): BERT-style masked
  prediction over music audio with **two teachers**: an RVQ-VAE (EnCodec) providing
  discretized *acoustic* targets, and a **CQT reconstruction target** injecting *pitch and
  harmonic* inductive bias. 95M and 330M variants, 24 kHz input, 75 Hz frames. SOTA-ish
  across 14 MIR tasks at release; the standard music representation for linear-probe studies.
- MusicFM, MULE: alternative music SSL encoders worth including in encoder comparisons.

**(c) Contrastive audio-text encoders.**
- **CLAP** (LAION-CLAP, MS-CLAP): dual-tower audio/text embedding trained on caption pairs,
  music variants exist. Gives *clip-level* semantics (good for genre/mood/retrieval), weak
  temporal and pitch detail. Used for zero-shot classification and as retrieval scaffolding.

**(d) Neural audio codecs (discrete tokens).**
- **SoundStream / EnCodec / DAC**: residual vector quantization (RVQ) → parallel streams of
  discrete tokens (~50–75 Hz × 4–32 codebooks) optimized for *reconstruction*, not meaning.
  These power music *generation* (MusicGen; MusicLM pairs semantic tokens from w2v-BERT with
  acoustic codec tokens) and fully-tokenized "omni" models. Codec tokens preserve acoustic
  detail but encode it in a form LLMs find hard to reason over symbolically — the
  semantic-vs-acoustic token tension is a live research thread (e.g., unified tokenizers
  like EntangleCodec, 2026).

## 1.3 The standard LALM recipe (encoder → bridge → LLM)

```
16 kHz waveform → mel/fbank → audio encoder → downsample/pool → bridge → frozen-ish LLM
                                                                  |
                                    (a) linear/MLP projector (LLaVA-style; AF3, Qwen-Audio)
                                    (b) window-level Q-Former (SALMONN)
                                    (c) cross-attention layers (original Flamingo, AF1/AF2)
```

- The **bridge/adapter** maps encoder vectors into the LLM's embedding space; audio arrives
  as a prefix of soft tokens (30 s ≈ 375–750 tokens depending on rate).
- Trend over 2024–2026: away from Q-Former/cross-attention toward simple projectors with
  *better data and curricula* doing the heavy lifting; and toward "omni" models where audio
  tokens share the causal decoder with text natively (Qwen3-Omni's Thinker-Talker MoE:
  Thinker does understanding/text, Talker streams speech out — Talker is irrelevant to
  understanding research).

## 1.4 How they're trained (the part that explains most failures)

Canonical multi-stage curriculum — AF3 is the cleanest fully-open documented example
(five stages):

1. **Encoder pretraining** (before LALM assembly): ASR supervision (Whisper/AuT) or SSL
   (MERT). *Determines what's perceivable at all.*
2. **Alignment pretraining**: freeze encoder + LLM, train only the adapter on
   audio-captioning / ASR / tagging data. Teaches the LLM "what audio tokens mean" at
   caption granularity.
3. **Encoder tuning**: unfreeze encoder (+adapter), broaden to diverse audio tasks.
4. **Full instruction tuning (SFT)**: unfreeze everything; large synthetic QA corpora —
   AF3 used **AudioSkills-XL (~8M QA pairs)**; long-audio stage adds LongAudio-XL (1M+ QA,
   up to 10 min).
5. **Post-training**: chain-of-thought data (AF-Think, 250K), multi-turn chat (AF-Chat), and
   increasingly **RL with verifiable rewards** (GRPO-style). **Music Flamingo** = AF3
   backbone + music-heavy continued training + theory-aware long captions + CoT + RL with
   custom music rewards.

**The critical detail — where the QA data comes from:** instruction pairs are overwhelmingly
*generated by text LLMs from metadata, tags, and captions* of audio datasets — not from
ground-truth musical analysis of the signal. Consequences:
- Supervision is caption-grained ("upbeat jazz with walking bass"), almost never
  note/interval/voicing-grained. Models are never *forced* to resolve precise pitch.
- Text-LLM priors get baked in: the model learns what captions plausibly say about audio
  like this, which is exactly the shortcut MuChoMusic caught models using (answering music
  MCQs decently *without the audio*).
- Skills follow data density: speech tasks ≫ sound events ≫ music semantics ≫ music theory.

## 1.5 Why music specifically is a stress test

1. **16 kHz resampling** throws away the top 1.5 octaves of content before encoding.
2. **ASR-shaped encoders**: objective never rewarded absolute pitch, tuning, voicing.
3. **Token-rate bottleneck**: fast events (ornaments, fast runs, ghost notes, strumming
   detail) compress into 1–2 vectors.
4. **Caption-level supervision**: no note-level ground truth in training.
5. **Text-prior shortcutting**: models answer plausibly without listening (MuChoMusic).
6. **Instruction-following gap on MIR tasks**: CMI-Bench reframed classic MIR tasks
   (key, beat, melody extraction) as instructions and found large gaps vs supervised models.
7. **Long-context degradation**: accuracy drops >10 points beyond ~5 minutes of audio —
   relevant for form/structure tasks.

Each of these is a *hypothesis generator* for Part II: they predict which tasks fail and at
which level (L1/L2/L3) the failure should show up.

## 1.6 State of the art (as of mid-2026)

**Open weights (probeable — our white-box targets):**
- **Music Flamingo** (NVIDIA, Nov 2025): open SOTA for music understanding; AF3 architecture
  (AF-Whisper + projector + Qwen2.5-7B), music-specialized with CoT + RL. Primary target.
- **Audio Flamingo 3** (NeurIPS 2025): fully open (weights + data recipe + code); the best
  *documented* pipeline, which makes it the best model to study and fine-tune.
- **Audio Flamingo Next / AF-Next** (Apr 2026): newest in the series; check weights/recipe
  availability during Phase 0.
- **Qwen3-Omni (30B-A3B MoE)** and **Qwen3.5-Omni** (Apr 2026): strongest general open
  omni-models; AuT encoder @ 12.5 Hz; MoE Thinker. Heavier to probe but important because
  the encoder is *not* Whisper-derived — a natural architecture contrast with AF3.
- **Qwen2-Audio-7B**: older, widely benchmarked; cheap comparison anchor.
- Standalone encoders for L2 work: **MERT-330M**, CLAP, Whisper-enc, AF-Whisper, AuT.

**Closed (behavioral ceilings via API):** Gemini 2.x (native audio). GPT-4o-audio is out of
scope (no OpenAI API access, decided 2026-07-25) — Gemini is the study's only closed-model
ceiling. (Claude takes no raw audio — but is useful for the symbolic-music text contrast:
ABC/MIDI-as-text.)

**Benchmark state:**
- **MMAU** (ICLR'25) and **MMAU-Pro** (Aug'25, 5.3K expert instances, 49 skills): humans
  77.9%, **no model above ~60%** on Pro. Music is consistently the weakest domain.
- **MuChoMusic** (ISMIR'24): the text-prior exposé; 1.1K validated MCQs.
- **CMI-Bench** (2025): music instruction-following with proper MIR metrics; LALMs far below
  supervised MIR baselines on key/beat/melody.
- **MARBLE**: encoder-level MIR benchmark suite (the L2 reference).
- Recent probing-flavored evals: "core music perception" batteries (2025) and automated
  perception testing on arbitrary music ("Music I Care About," 2026) — closest neighbors to
  our Track A; read before building the harness to avoid duplication and to steal design.

**The headline picture**: models are good at *describing* music (genre, mood,
instrumentation, captions) and bad at *hearing* music precisely (key, intervals, chord
quality, beat structure, technique detail) — i.e., strong caption-space semantics, weak
grounded perception. Our study's job is to turn that folk summary into a mechanistic,
layer-located, causally-tested account.

## 1.7 Reading list (in order)

*Foundations*
1. Whisper paper (Radford et al., 2022) — the default encoder's objective and biases.
2. HuBERT (2021) — masked-prediction SSL; the template MERT follows.
3. **MERT** (ICLR'24) — music SSL with RVQ-VAE acoustic + CQT musical teachers.
4. CLAP (2023) — contrastive audio-text; zero-shot classification.
5. EnCodec (2022) + MusicGen (2023) — codec tokens; the generation-side view.

*Audio LLMs*
6. Qwen2-Audio (2024) — minimal canonical LALM recipe.
7. SALMONN (ICLR'24) — Q-Former bridging, dual encoder.
8. Audio Flamingo 2 (2025) → **Audio Flamingo 3** (NeurIPS'25) — read AF3 closely: 5-stage
   curriculum, AF-Whisper, AudioSkills-XL data generation. This is the recipe you'd adapt
   to train your own model.
9. **Music Flamingo** (2025) — music specialization: data, CoT, RL rewards.
10. Qwen3-Omni technical report (2025) + Qwen3.5-Omni (2026) — AuT encoder, MoE omni design.

*Evaluation & analysis*
11. MuChoMusic (ISMIR'24) — text-prior problem + MCQ methodology.
12. MMAU (ICLR'25), MMAU-Pro (2025) — broad audio reasoning; music subscores.
13. CMI-Bench (2025) — MIR-metric instruction following.
14. MARBLE (2023) — encoder probing benchmark.
15. Surveys: "Towards Holistic Evaluation of LALMs" (2505.15957) and "A Survey of Large
    Audio Language Models: Generalization, Trustworthiness, Outlook" (2605.20266).
16. Recent perception batteries: arXiv 2510.22455 (core music perception in MLLMs),
    2607.06015 (Music I Care About), 2511.05550 (factual music comprehension).

---

# PART II — THE STUDY

## 2. Research questions

**RQ1 (Capability map).** Across *all genres*, which music understanding tasks can current
models perform, at what precision? Low-level perception (notes, intervals, tuning, tempo) →
mid-level structure (key, scale/mode incl. blues/dorian/phrygian, chords, instrument and
vocal technique) → high-level semantics (genre/subgenre, form, production style, mood).

**RQ2 (Mechanism).** Where does musical information live in the pipeline? Is it present in
the encoder representations, and does the LLM *use* it — or answer from text priors and
coarse timbral statistics? How do representations differ across encoder families
(ASR-supervised Whisper/AuT vs music-SSL MERT vs contrastive CLAP)?

**RQ3 (Failure etiology).** For each observed failure, which cause is it?
- **(a) Data coverage** — the skill is learnable by this architecture but undertrained
  (fixable with instruction data);
- **(b) Alignment gap** — the encoder represents it but the LLM can't verbalize it
  (fixable with cheap adapter/LLM-side tuning);
- **(c) Architectural bottleneck** — 16 kHz input, ASR-biased encoder, or token rate
  destroys the information before the LLM sees it (needs encoder/front-end changes).

**RQ4 (Actionable).** What does RQ1–3 imply for training a music understanding model:
which encoder, what token rate, what data mix, which training stage to intervene at?

*(Cultural generalization — e.g., Indian classical, maqam, gamelan — is deferred to an
optional extension module, §8. The machinery built here answers it later for free.)*

### The central logic: a 3-level dissociation

For every property, measure at three levels on the **same stimuli**:

| Level | Measurement | Interpretation when this succeeds but the next fails |
|---|---|---|
| **L1: Signal** | Classical MIR / small supervised model extracts the property | property is physically recoverable from the audio |
| **L2: Representation** | Linear probe on encoder / LLM hidden states decodes it | info is *encoded* but not *verbalized* → alignment gap (RQ3-b) |
| **L3: Behavior** | The LALM answers questions about it | — |

- L2 high / L3 low → alignment gap → cheap fine-tuning fix (confirmed causally in Track C).
- L2 low / L1 high → encoder never learned it → pretraining data or architecture (RQ3-a/c);
  disambiguate with the token-rate and encoder-family analyses (§5.3–5.4).
- L3 high / L2 low → red flag: behavioral success via priors/shortcuts, not listening.

## 3. Task taxonomy — the probe battery

Every experiment is a cell in: **task × stimulus source (synthetic / real) × genre ×
measurement level (L1/L2/L3)**.

### Tier 1 — Low-level perception (synthesizable → perfect ground truth)
1. Pitch: single-note ID (absolute; and relative given a reference), octave placement.
2. Intervals: name the interval; ascending/descending; melodic vs harmonic.
3. **Microtonal discrimination**: higher/lower/same at Δ ∈ {5,10,25,50,100} cents →
   a psychometric curve per model. Also: in-tune vs detuned (quarter-tone flat) note in a
   chord; overall tuning reference (A=440 vs 432).
4. Tempo (BPM estimation, error distribution — octave errors are diagnostic); meter
   (3/4 vs 4/4 vs 6/8 vs 7/8); swing vs straight; syncopation detection.
5. Timbre: instrument ID across families (NSynth) and in-mix (MedleyDB stems).
6. Polyphony: count simultaneous notes; count instruments in a mix.
7. Dynamics & articulation: staccato/legato, crescendo, accent placement.

### Tier 2 — Mid-level structure
8. Key identification (all 24 major/minor), on scales, chord progressions, and real mixes.
9. **Scale/mode identification**: major, natural/harmonic/melodic minor, dorian, phrygian,
   lydian, mixolydian, locrian, blues scale, major/minor pentatonic, whole-tone,
   diminished — presented as (a) bare scales, (b) melodies *in* the mode over a drone,
   (c) full arrangements. (b) and (c) are the diagnostic ones.
10. Chords: quality (maj/min/dim/aug/7ths/extensions), inversions; progressions
    (I–IV–V–I vs I–V–vi–IV vs 12-bar blues vs ii–V–I); modulation detection.
11. **Instrument technique**: guitar — bend, slide, hammer-on/pull-off, palm mute, natural
    harmonics, fingerpicking vs plectrum (GuitarSet, IDMT-SMT-Guitar); strings — pizzicato/
    arco/tremolo/col legno; drums — ghost notes, rimshot, brushes; piano — pedaling.
12. **Vocal technique**: vibrato (and its rate/extent), straight tone, belt, breathy, vocal
    fry, trill, falsetto vs head voice, melisma, growl/scream (metal), rap flow vs sung —
    VocalSet (17 techniques × 20 singers) is purpose-built for the core of this.
13. Continuous-pitch ornament perception across styles: blues bends & blue notes, country
    slides, jazz scoops/falls, gospel runs, pitch-corrected (Auto-Tune) vs natural vocals.
14. Melody extraction/matching: same melody, different arrangement — recognized or not?

### Tier 3 — High-level / semantic (the all-genre layer)
15. Genre & **subgenre** classification: coarse (GTZAN-style 10-way, with caution — known
    label noise) → fine (MTG-Jamendo tags; metal subgenres, EDM subgenres, jazz eras,
    hip-hop regional styles). Include a **genre-cue ablation**: re-render the same piece
    (MIDI → different soundfonts/production) to test whether genre judgments track
    composition or production texture.
16. Era/production style: decade estimation; production techniques — sidechain compression,
    distortion type, reverb size, lo-fi artifacts, sampling vs live drums.
17. Form/structure: verse/chorus/bridge; 12-bar blues cycle; AABA; drop detection (EDM);
    call-and-response; where does the section boundary fall (long-context stress test).
18. Mood/emotion; music captioning quality (vs MusicCaps references); lyric-audio
    integration (does the model use sung lyrics as a shortcut for genre/mood?).
19. Similarity/cover-song: same song, different artist/genre rendition.

## 3.5 Model roster — the experiments per model, and how each is run

This section makes the study concrete: every model we test, which experiments run on it,
and where/how it physically runs. (Tracks are defined in §4–6; task numbers refer to §3.)

### At a glance

| Model | Access | Track A (behavior) | Track B (probing) | Track C (LoRA) | Runs on |
|---|---|---|---|---|---|
| **Music Flamingo** | open weights | full battery | full layer-wise | secondary check | Colab A100, bf16 |
| **Audio Flamingo 3** | open (weights+code+data) | full battery | full layer-wise | **primary platform** | Colab A100, bf16 |
| AF-Next | check in Phase 0 | battery if open | optional | — | TBD |
| Qwen3-Omni-30B-A3B | open weights (MoE) | full battery | stretch goal | — | Colab A100, 8-bit |
| Qwen2.5-Omni-7B | open weights | Tier 1–2 anchor | full layer-wise | — | Colab A100, bf16 |
| Qwen2-Audio-7B | open weights | battery (anchor) | — | — | Colab L4/A100 |
| Gemini 2.x | API only | full battery + long-context | — | — | local harness → API |
| ~~GPT-4o-audio~~ | ~~API only~~ | **out of scope — no OpenAI API access (2026-07-25)** | — | — | — |
| Claude | API, no audio in | symbolic contrast only | — | — | local harness → API |
| MERT-330M (+95M) | open encoder | — | full probe suite | — | Colab L4 |
| CLAP (LAION/MS) | open encoder | zero-shot classification | probe suite | — | Colab L4 |
| Whisper-large-v3 encoder | open encoder | — | probe suite | — | Colab L4 |
| AF-Whisper | open encoder | — | probe suite | — | Colab L4 |
| AuT (from Qwen3-Omni) | open, if extractable | — | probe suite | — | Colab A100 |

Every Track A run uses the same manifest, prompts, controls, temperature 0, and logging
(§4 logistics); every Track B run uses the same probe suite and leakage-safe splits (§5);
so all cells of the table are directly comparable.

### White-box LALMs (the models we can open up)

**Music Flamingo — the "best case" open model.**
- *Track A:* the entire §3 battery, Tiers 1–3, all seven controlled contrasts. As the open
  SOTA for music, its failures are the interesting ones: anything Music Flamingo can't do,
  weaker models won't either.
- *Track B:* layer-wise linear probes at three tap points — AF-Whisper encoder layers,
  post-adapter (what the LLM actually receives), and LLM decoder layers at audio positions.
  Probe targets: pitch class, octave, interval, key, mode, chord quality, instrument,
  technique, tempo/meter, genre.
- *Track C:* used as a replication check for whichever arm wins on AF3 (does the same LoRA
  recipe close the same gap on the music-specialized model?).

**Audio Flamingo 3 — the workhorse.**
- *Track A:* full battery. Comparing AF3 vs Music Flamingo isolates what music-heavy
  post-training buys, since they share architecture.
- *Track B:* full layer-wise probing, same tap points as Music Flamingo.
- *Track C:* **all intervention experiments run here first** — it's the only model with
  fully open training code + data recipe. Three arms per selected skill: (i) LLM-side LoRA,
  (ii) LLM + encoder-side adapters, (iii) matched-size unrelated-audio control. Evaluated
  on held-out soundfonts/singers plus a synthetic→real transfer set.

**AF-Next.** Phase 0 action item: check weight/recipe availability. If open, it gets the
Track A battery (is the newest generation better at perception, or just at description?);
Track B only if its architecture is close enough to AF3 to reuse the extraction code.

**Qwen3-Omni-30B-A3B — the architecture contrast.** Its AuT encoder is trained from
scratch (not Whisper-derived) at 12.5 Hz, making it the key test of "is everything
downstream of Whisper's biases?"
- *Track A:* full battery.
- *Track B:* stretch goal only — 30B MoE needs 8-bit quantization and patience; if
  activation extraction proves too painful, Qwen2.5-Omni-7B substitutes (below).

**Qwen2.5-Omni-7B — the practical non-Whisper representative.**
- *Track A:* Tier 1–2 battery (enough to anchor its behavior against its probes).
- *Track B:* full layer-wise probing — this is where the non-Whisper-encoder comparison
  actually happens at manageable cost.

**Qwen2-Audio-7B — the cheap anchor.**
- *Track A only:* full battery. Two purposes: (1) Phase 0 harness validation — replicate
  its published MMAU-music / MuChoMusic numbers to prove our plumbing is right before
  trusting any novel result; (2) a widely-benchmarked reference point connecting our
  numbers to the literature.

### Black-box API models (behavioral ceilings)

**Gemini 2.x (native audio).** (GPT-4o-audio dropped 2026-07-25 — no OpenAI API access;
Gemini is the study's only closed-model ceiling.)
- *Track A only:* full battery through the same harness (audio + prompt via API,
  temperature 0, everything logged). Gemini additionally gets the long-audio structure
  tasks (task 17) since its context window permits >5 min clips. Caveats: API audio-length
  and format limits get documented per model in Phase 0; no internals means no Track B/C —
  these models give us the field's behavioral ceiling, not mechanism.

**Claude (no raw audio input).**
- *Symbolic contrast arm only* (§4, contrast 6): the same Tier 1–2 questions posed as
  ABC-notation/MIDI-as-text. This measures pure symbolic music knowledge with no listening
  involved — the "knows music theory but can't hear" endpoint that calibrates how much of
  other models' scores could come from theory knowledge alone. The audio-capable models
  get the same symbolic variants, so we can compare each model's audio vs symbolic
  performance head-to-head.

### Standalone encoders (Track B only — the L2 comparison set)

These never answer questions; we probe their representations directly. Same stimuli, same
probe suite, same splits — this is the encoder-family comparison of §5.2, and it's the
cheapest, highest-value part of Track B (all run on a modest Colab GPU in hours).

- **MERT-330M** (and 95M for a size check): the music-SSL reference — expected L2 upper
  bound. Also the substrate for the geometry analyses (§5.3: pitch helix, 12-TET
  quantization) and the 75 Hz side of the temporal-resolution contrast (§5.4).
- **Whisper-large-v3 encoder**: the ASR-shaped baseline — what most LALMs actually hear
  through.
- **AF-Whisper**: Whisper after NVIDIA's audio/music continued pretraining. Probing it vs
  vanilla Whisper isolates exactly what that continued pretraining added.
- **CLAP**: the contrastive clip-level representative — expected strong on genre/mood,
  weak on pitch/temporal targets. Its zero-shot classification mode also serves as a
  simple no-LLM behavioral baseline on Tier 3 tasks.
- **AuT** (if extractable from Qwen3-Omni): the from-scratch ASR encoder at 12.5 Hz — the
  low-frame-rate side of the temporal-resolution contrast against MERT's 75 Hz (hypothesis
  H5's decisive test).

### How each kind of run physically works

- **API models**: the harness runs on the local machine; each manifest row → one API call
  (audio attachment + prompt) → response logged to the results table. Costs money per
  call; no GPU needed.
- **Open 7B-class LALMs**: Colab Pro A100-40GB, bf16, batch 1–4. Track A = generate
  answers; Track B = forward passes with hidden-state hooks, activations saved to Drive as
  fp16, probing done offline with scikit-learn on any machine.
- **Qwen3-Omni-30B MoE**: same, but 8-bit quantized; treat every run as expensive —
  batch the manifest, checkpoint progress to Drive.
- **Standalone encoders**: Colab L4 (or even CPU for CLAP) — trivial compute.
- **Track C LoRA runs**: Colab A100 with `peft`; adapters + eval outputs checkpointed to
  Drive each session; per-skill datasets are 5–20k QA pairs, so single-session training
  runs are feasible.

### Model × phase map

- **Phase 0**: Qwen2-Audio-7B (harness validation); AF3 or Music Flamingo running on
  Colab; AF-Next availability check.
- **Phase 1** (synthetic Tier 1–2, Track A): Music Flamingo, AF3, Qwen3-Omni + Gemini;
  Claude symbolic arm. (GPT-4o-audio dropped 2026-07-25 — no OpenAI API access.)
- **Phase 2** (real-music Tier 3 + contrasts): same roster.
- **Phase 3** (Track B): encoder set (MERT, CLAP, Whisper-enc, AF-Whisper, AuT) +
  layer-wise probes on AF3 or Music Flamingo + Qwen2.5-Omni-7B.
- **Phase 4** (Track C): AF3 (all arms), Music Flamingo (replication of winning arm).

## 4. Track A — Behavioral probing (black-box, all models)

**Design principles**
- **Synthesize wherever possible**: MIDI + fluidsynth (multiple soundfonts) for pitch/key/
  mode/chords/tempo with perfect ground truth and orthogonal factor control (pitch ×
  instrument × register × tempo). Microtonal stimuli via pitch-bend or direct synthesis;
  WORLD-vocoder resynthesis to manipulate vocal pitch contours holding timbre constant.
- **Real audio where semantics matter**: genre/production/structure tasks need real mixes —
  MTG-Jamendo, MedleyDB, FMA; freshly-recorded or post-cutoff releases for
  contamination-sensitive claims.
- **MCQ + open-ended, always both**; text-LLM judge for open-ended, validated against ~100
  hand-scored items.
- **Controls on every task** (non-negotiable, per MuChoMusic):
  - *No-audio control*: same question, no audio. If accuracy barely drops, the item measures
    text priors — flag or discard.
  - *Silence / noise / wrong-audio* controls on a subsample.
  - *Choice-order & label permutation*; ≥3 paraphrases per question type.
- **Psychometrics over accuracy**: cents-discrimination thresholds; key confusion matrices
  (circle-of-fifths-adjacent and relative-major/minor errors = real listening; uniform
  errors = guessing); tempo octave-error rates; technique confusions (bend↔slide is an
  honest perceptual error; bend↔palm-mute is not).

**Key controlled contrasts** (each isolates one candidate cause):

| Contrast | Isolates |
|---|---|
| Same progression: piano vs distorted guitar vs synth pad | harmony perception vs timbre |
| Same piece: original mix vs MIDI re-render vs stems-removed | genre = composition vs production texture |
| Mode ID: bare scale vs melody-in-mode vs full arrangement | pitch-set matching vs contextual tonality |
| Transposition ±1–5 semitones | absolute vs relative pitch encoding |
| Natural vs Auto-Tuned vocal, same phrase | continuous-pitch perception |
| Same question: audio vs ABC-notation-as-text | acoustic perception vs symbolic music knowledge |
| Sung lyrics intact vs instrumental version | lyric shortcut in genre/mood answers |

**Logistics.** One manifest-driven eval harness (`stimuli.parquet`: path, task, ground truth,
factors, controls; every answer logged with prompt, seed, temperature=0). Runs locally for
API models, on Colab for open models. 50–200 items/cell; breadth over depth in Phase 1.

## 5. Track B — Representational probing (white-box, open models)

1. **Layer-wise linear probes** on (a) standalone encoders (MERT, CLAP, Whisper-enc,
   AF-Whisper, AuT if extractable) and (b) full LALMs (encoder layers, post-adapter, LLM
   decoder layers at audio positions). Probe targets: pitch class, octave, interval, key,
   mode, chord quality, instrument, technique, tempo/meter, genre. → Probe-accuracy-by-layer
   curves per property.
2. **Encoder-family comparison** (RQ2's sharpest tool): the same probe suite across
   ASR-supervised (Whisper/AuT) vs music-SSL (MERT) vs contrastive (CLAP) encoders
   quantifies what each pretraining objective preserves — directly informs encoder choice
   for your own model.
3. **Geometry & quantization probes**: does pitch form a helix/circle? Is representation
   quantized toward 12-TET (probe error vs distance-from-nearest-semitone — scalloped error
   = 12-TET bias)? Is key encoded absolutely or relatively (train probe transposed, test
   untransposed)?
4. **Temporal-resolution analysis** (RQ3-c): probe for within-frame pitch-contour slope and
   extent; compare properties that live above vs below the token rate (sustained key vs
   fast ornament). If ornament information is unrecoverable at 12.5 Hz but recoverable at
   75 Hz (MERT), that's an architectural answer.
5. **The L2/L3 dissociation table**: probe accuracy vs behavioral accuracy per
   (property × genre) — the quadrant analysis from §2.
6. **Attention/attribution (lightweight)**: do answer tokens attend to the relevant audio
   frames (the ornamented note, the modulation point) or uniformly?

**Colab feasibility:** MERT/CLAP probing is trivial (hours on L4). AF3/Music-Flamingo-7B
activation extraction in bf16 on A100-40GB, batch 1–4, cache activations to Drive as fp16,
probe offline with scikit-learn. Qwen3-Omni MoE-30B needs 8-bit + patience; treat as
stretch goal or use Qwen2.5-Omni-7B as the non-Whisper-encoder representative.

## 6. Track C — Causal intervention (the data-vs-architecture test)

Pick the 3–5 clearest **L2-high/L3-low** failures from Tracks A+B (predicted candidates:
key ID, mode ID, chord quality, technique naming). Then:

1. **Build small instruction sets per skill** (5–20k QA triples each), generated
   programmatically: synthetic MIDI renders for key/mode/chords; VocalSet/GuitarSet labels
   for technique. Zero manual annotation.
2. **Three-arm LoRA fine-tune** on AF3 (best-documented training code):
   - (i) LLM-side LoRA, encoder frozen → tests the alignment-gap hypothesis;
   - (ii) + encoder-side adapters → tests the encoder-gap hypothesis;
   - (iii) control arm tuned on matched-size unrelated audio QA → controls for generic gains.
3. **Evaluate on held-out** instruments/singers/soundfonts (not just held-out clips —
   guard against timbre/artist leakage), plus a *transfer* set (real mixes for a skill
   trained on synthetic stimuli).
4. Interpretation: (i) closes the gap → alignment-only, data-fixable (RQ3-b). Only (ii)
   helps → encoder gap (RQ3-a). Neither, and Track B shows the info missing at all layers →
   architectural (RQ3-c), with the token-rate/encoder-family evidence saying what to change.

This is also a dry run of exactly the fine-tuning stack you'd use to train your own model.

## 7. Datasets & tooling

**Stimuli/data:** synthetic (pretty_midi + fluidsynth, multiple soundfonts; numpy/DDSP for
microtonal; WORLD for vocal resynthesis), NSynth (instruments/pitch), VocalSet (vocal
technique), GuitarSet + IDMT-SMT-Guitar (guitar technique), MedleyDB (stems/melody),
MTG-Jamendo (tags/genre at scale), FMA, GTZAN (legacy comparison only), MusicCaps
(captioning reference), Groove MIDI (drums/microtiming).

**Software:** `mirdata`, `librosa`, `essentia`, `pretty_midi`, `crepe`/`pyin` (L1 pitch),
`madmom` (L1 beat/key), `transformers` + `peft` (models, LoRA), `scikit-learn` (probes),
custom manifest-driven eval harness (first artifact to build — well-suited to building with
Claude Code).

**Compute split:** local/Claude Code = stimulus synthesis, API evals, judging, analysis,
plots. Colab Pro = open-model inference, activation extraction, LoRA training (checkpoint
to Drive; A100 sessions).

## 8. Optional extension module — cultural generalization (deferred)

The same L1/L2/L3 + contrast machinery, applied to traditions with non-12-TET intonation
and different structural grammars: Indian classical (Saraga dataset; tonic/raga/tala/gamaka
tasks), Arabic maqam, gamelan, flamenco. Deferred, not deleted: Tier-1 microtonal
psychometrics (task 3) and the technique/ornament battery (tasks 11–13) already collect the
perceptual prerequisites, so this module later becomes a pure add-on. Full design notes preserved in
`archive/carnatic-module-notes.md`.

## 9. Pitfalls checklist

- [ ] Text-prior shortcutting → no-audio controls on every task.
- [ ] MCQ position/label bias → permute; report bias-corrected accuracy.
- [ ] **Training-data contamination**: GTZAN, MusicCaps, FMA are very likely in these
      models' training data. For high-stakes claims use synthetic stimuli, fresh recordings,
      or post-2026 releases; consider audio fingerprint near-dup checks.
- [ ] Timbre/artist/session leakage in probe train/test splits (split by singer, by
      soundfont, by recording session).
- [ ] Judge-model error on open-ended answers → validate against hand-scored sample.
- [ ] Loudness/duration/quality mismatches across stimulus arms → normalize.
- [ ] Over-reading probes: linear decodability ≠ causal use — that's why Track A (behavior)
      and Track C (intervention) bracket Track B from both sides.
- [ ] Version-pin models; log everything (model rev, prompt, audio hash, sampling params).

## 10. Phased execution

**Phase 0 (wk 1–2): Foundations + infrastructure.**
Work through reading list §1.7 (at least items 3, 8, 9, 11, 13). Build the eval harness +
stimulus manifest schema. Get AF3 or Music Flamingo running on Colab. Replicate 2–3
published MMAU-music / MuChoMusic numbers to validate the harness. Check AF-Next weight
availability.

**Phase 1 (wk 3–6): Synthetic battery — Tier 1–2, Track A.**
Pitch, intervals, cents psychometrics, tempo/meter, key, modes, chords, techniques
(VocalSet/GuitarSet). 2–3 open models + 1–2 API models.
*Deliverable: capability heatmap (task × model) + psychometric curves.*

**Phase 2 (wk 6–9): Real-music & all-genre battery — Tier 3 + contrasts, Track A.**
Genre/subgenre with cue-ablation, production style, structure, the seven controlled
contrasts of §4.
*Deliverable: what "genre understanding" is actually made of; contrast-controlled failure list.*

**Phase 3 (wk 8–12, overlapping): Track B.**
Encoder-family probe comparison; layer-wise probes on one full LALM; 12-TET quantization +
temporal-resolution analyses; the L2/L3 dissociation table.
*Deliverable: "where each skill lives / dies" diagnosis.*

**Phase 4 (wk 12–16): Track C.**
Three-arm LoRA on the top L2-high/L3-low failures.
*Deliverable: causal data-vs-architecture verdict per skill + a validated fine-tuning recipe.*

**Phase 5: Write-up + (optional) cultural-generalization module (§8).**
The dissociation table + intervention results is a paper (ISMIR / NeurIPS D&B / ICASSP);
the harness + instruction sets are releasable artifacts.

## 11. Hypotheses (stated up front)

*Status column added 2026-07-22 after Track A (6/7 models) + Track B (probes, partial
attention) landed. Full evidence in PAPER.md § Results; this is a one-line pointer per
hypothesis, not the analysis.*

- **H1**: Description ≫ perception — genre/mood/instrumentation strong; key, chord quality,
  interval naming weak, across all models (predicted by §1.4's caption-grained supervision).
  — **Untested**: Tier 3 (genre/description) battery not run yet; only Tier 1–2 exists.
  Key/chord/interval weakness on the perception side is already confirmed on open-format
  responses (near-floor for most models), so half the comparison is in.
- **H2**: Cents-discrimination thresholds coarse (≥50 cents) for LALMs; MERT probes will
  beat LALM behavior on fine pitch (predicted by ASR encoders + token rate, §1.5).
  — **Partially supported**: MERT beats Whisper-enc on `pitch_note_id` (74% vs 45%,
  chance 8%) and `key_id` (21% vs 12%, chance 4%), consistent with its CQT-reconstruction
  pretraining target. Formal cents psychometric curve (threshold in cents) not plotted yet.
- **H3**: Genre classification substantially survives no-audio and lyric shortcuts but
  degrades sharply under MIDI re-render → genre skill is largely production-texture +
  prior-driven, not compositional.
  — **Untested**: Tier 3 not run.
- **H4**: Key/mode/chord info decodable from mid encoder layers (L2 high) while behavior
  fails (L3 low) → alignment gap → Track C arm (i) recovers most of it.
  — **Mixed**: the pattern shows up clearly, but not on key/mode/chord — it's
  `beats_per_bar`, `octave_id`, `tuning_judgment`, `cents_discrimination`, `note_count`
  that are L2-high/L3-low. Key/mode/chord instead show L3 > generic-encoder L2, which is
  the *inverse* of H4 and needs re-probing the LALMs' own encoders before drawing a
  conclusion either way (see PROJECT_STATE next actions). `beats_per_bar` additionally
  fails a wrong-audio control, so it may not be a real alignment gap at all — see PAPER.md.
- **H5**: Fast-ornament and technique detail unrecoverable even by probes at 12.5 Hz token
  rates but partially recoverable at MERT's 75 Hz → architectural bottleneck class.
  — **Untested**: no ornament/technique tasks in the battery yet (Tier 2 items 11–13,
  VocalSet/GuitarSet, not implemented); AuT (12.5 Hz) not yet extracted/probed.
- **H6**: Music Flamingo > AF3 > Qwen-Omni on music behavior, but encoder probes (L2) will
  be far more similar than behavior (L3) across models — most variance lives in alignment
  and instruction data, not perception.
  — **Supported on the ordering, untested on the L2-similarity claim**: Music-Flamingo
  does lead on `key_id` (75%) and shows the strongest wrong-audio-control drop (+10.8pp,
  best evidence of real listening among the 6). The L2 side of this hypothesis compares
  standalone generic encoders (MERT/Whisper/CLAP), not each model's own encoder, so it
  can't yet be checked directly — same re-probing gap as H4.
- **H7** (added 2026-07-24, Track D): a second input modality carrying the same information
  at the same abstraction level as audio (a spectrogram rendered as an image) will measure
  differently than a second modality that pre-computes the answer (rendered sheet music) —
  the former is a fair test of whether vision-pretrained pathways extract musical structure
  better than audio-pretrained ones; the latter is answer leakage, not a perception test.
  Prediction: fine-tuned audio+spectrogram-image beats audio-only on the L2-high/L3-low
  shortlist tasks, *and* the wrong-image control shows a real drop (not near-zero, the way
  `beats_per_bar`'s wrong-audio delta was) — if the control is flat, the model learned to
  read the image and ignore the audio, which is a null result dressed as a positive one.
  — **Resolved, mixed (Tracks D/G, 2026-07-26 to 08-05)**: a plain spectrogram (same
  abstraction level as audio) was a precise null on the pitch/tuning shortlist — the
  model ignored it even when forced to rely on it (Track D conclusive/force). What
  worked was neither a same-level spectrogram nor answer-leaking sheet music, but a
  *zoomed, reference-annotated* chart (Track D-zoom: cents 0.55→0.94, tuning 0.53→0.89)
  — a third category H7 didn't anticipate: same-level information, but rendered with an
  explicit, non-leaking annotation the model can read a position off of. Track G tested
  only the *first* of Track D's stages on the harmonic cluster — a flat, unzoomed,
  unannotated chromagram, force-trained (modality dropout, same as `train_track_d_force.py`)
  — and it was a clean null on `key_id`/`mode_id`/`chord_quality`/`interval_id` (all 95%
  CIs include 0), with the mechanism check showing the dropout training didn't even force
  reliance the way Track D's force stage did (wrong-chromagram ≈ no-chromagram on 3 of 4
  tasks, unlike Track D force's wrong-image-craters-performance result). **The zoom/
  annotation step that actually rescued pitch has NOT been tried on harmony** — don't read
  Track G as ruling out charts for this cluster the way the full D→D-zoom arc ruled things
  in for pitch; it only rules out the crude version. It's also a genuinely harder design
  problem here: D-zoom's reference line marked a task-*independent* value (the 12-TET
  in-tune grid), not the answer; the harmonic-cluster analogue (e.g. marking the tonic
  pitch-class row for `key_id`) risks marking the answer itself — the same leakage trap
  §12.2 already flagged for rendered sheet music. See PROJECT_STATE next action 13 for the
  ordered follow-up plan (zoom first, since it's leakage-safe; annotation only where a
  non-leaking design exists).

---

# PART III — TRACK D (planned, added 2026-07-24)

## 12. Track D — Multimodal representation exploration

**Where this comes from.** Decision 11 (§ PROJECT_STATE.md) reframed the end goal from "close
individual task gaps" to "find a candidate *universal* music representation" — one usable as
a training target for our own model. Five requirements were set for any candidate:

1. inferable from audio,
2. compact in tokens,
3. sufficient for the battery tasks (§3),
4. human-readable,
5. **genre-universal, including continuous pitch** (gamakas, blues bends, maqam) — staff
   notation and standard MIDI both fail this (pitch-bend is a hack), which is why the
   deferred Carnatic module (§8) is this requirement's eventual stress test, not an
   afterthought.

Track D asks a question the ladder idea (§0.7 rung 2, Part 0) didn't cover: not just *which
text features* to hand the model, but *which additional input modality* — and, separately,
whether a second modality is more useful as a training-time signal that shapes a better
internal representation than as a test-time crutch. It reuses Track A's stimulus/hygiene
infrastructure and Track B's probing infrastructure rather than building new ones.

### 12.1 Model-support audit (done 2026-07-24 — gates everything else)

Before building anything, checked which models can even take audio + image in one turn.
This cuts the candidate list hard:

| Model | Audio + image, same turn? | Source |
|---|---|---|
| Qwen2.5-Omni-7B | **Yes** — omni-modal by design (Thinker processes text/image/audio/video jointly) | [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni), [HF model card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) |
| Qwen3-Omni-30B-A3B | **Yes** — "understanding text, audio, images, and video" | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [technical report](https://arxiv.org/abs/2509.17765) |
| Gemini-2.5-Pro | **Yes** — single `generateContent` call accepts a mixed parts array (text/image/audio/video) | [Gemini API models](https://ai.google.dev/gemini-api/docs/models) |
| Qwen2-Audio-7B-Instruct | **No** — audio + text only | [HF model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| Audio-Flamingo-3 | **No** — AF-Whisper encoder + Qwen2.5-7B LLM, audio + text only, no vision tower | [HF model card](https://huggingface.co/nvidia/audio-flamingo-3-hf) |
| Music-Flamingo | **No** — explicitly audio (WAV/MP3/FLAC) + text only | [HF model card](https://huggingface.co/nvidia/music-flamingo-2601-hf) |

**Consequence that changes the plan**: AF3 and Music-Flamingo — the two best-documented,
most music-capable open models, and Track C's target — **cannot run Track D at all**. The
only open-weights candidate with full Track B/C infrastructure already built and at
manageable size is **Qwen2.5-Omni-7B**. Qwen3-Omni-30B-A3B stays a stretch goal for the same
reason it already was one in §3.5 (30B MoE, heavier to fine-tune/probe). Gemini-2.5-Pro can
run the Phase 1 accuracy question (black-box, no fine-tuning, no probing) as a ceiling
reference, same role it plays in Track A.

### 12.2 Phase 1 — does a spectrogram image, properly controlled, change accuracy?

**The trap to avoid.** Rendering *sheet music* from the same MIDI used to generate the audio
and feeding it as an input image is not a perception test — it's answer leakage. A rendered
score's key signature *is* `key_id`'s answer; its time signature *is* `beats_per_bar`'s;
notehead position *is* `pitch_note_id`/`octave_id`'s. This is the exact violation
`RUPALI_READ_THIS.md` rule 1 already warns about ("features must sit one abstraction level
below the answer") — sheet music sits *at* the answer. **Do not build this as a test-time
input.** (It comes back in Phase 2, as a training-time target only — see 12.3.)

**What's sound instead**: a **spectrogram rendered as an image**. Same abstraction level as
the audio itself (a re-rendering of the signal, not a symbolic answer), so it doesn't leak.
The hypothesis worth testing: these models' *audio* encoders were pretrained on
speech/music-specific objectives that never had to be good at reading 2D structure, while
their *vision* encoders were separately pretrained on huge general image data that's very
good at exactly that — edges, periodic textures, horizontal/vertical bands. A spectrogram
turns "hear the pitch" into "read a horizontal line's height," which the vision pathway may
simply be better at.

**Design:**
1. **Stimuli**: render one spectrogram PNG per existing manifest stimulus (deterministic,
   reuses `audio_path` — no new audio needed). Log-mel, matching what these models already
   compute internally as their own audio front-end (§0.2), so the image isn't handing the
   model a *different* transform, just the same one in a different modality.
2. **Hygiene layer, mirroring `jobs.py`'s existing pattern exactly**: add `image` /
   `no_image` / `wrong_image` conditions, with audio held at `audio` (correct) throughout —
   this experiment is specifically about what the image *adds*, not re-testing audio. Wire
   `wrong_image` the same way `wrong_audio` already works (`jobs.py:67-68`): drawn uniformly
   from the whole battery's spectrogram set, not filtered to the same task, scored against
   the *original* question's ground truth. Without this control, a positive result is
   uninterpretable — it could mean "the model is genuinely fusing image+audio" or "the model
   learned to just read the image and stopped using audio at all," and only the wrong-image
   condition tells them apart (same logic as H7 above and the existing wrong-audio
   discussion in the 2026-07-24 report correction).
3. **Zero-shot first, but don't trust a null result from it.** Prompting an off-the-shelf
   model with an audio+spectrogram-image pair it never saw paired in training is a weak test
   — a null could mean "doesn't help" or "doesn't know how to combine these." Plan for a
   LoRA fine-tune arm on Qwen2.5-Omni-7B (reusing Track C's training stack) from the start:
   train on (audio, spectrogram-image, question, answer) tuples from a training split, eval
   on held-out stimuli, same held-out-instrument/soundfont discipline as Track C (§6.3).
4. **Probe before and after** (Track B infrastructure, `extract_activations.py
   --own-encoder`): does joint training change what the *audio* encoder itself represents,
   or does the model route around it (learns to lean on the image pathway, audio
   representation unchanged or degraded)? This is the "internally" half of the user's
   question — accuracy alone can't answer it.
5. **Target tasks**: start with the tasks where behavior already trails a high L2 probe
   score (the existing alignment-fixable shortlist: `octave_id`, `tuning_judgment`,
   `cents_discrimination`, `note_count`) — if a spectrogram image doesn't move the needle
   there, where the information is already known to be extractable from *audio alone*, it's
   unlikely to be the fix. `key_id`/`mode_id`/`chord_quality`/`interval_id` are a secondary,
   noisier target given the own-encoder re-probe's read that behavioral success there is
   more likely priors than perception (2026-07-24 report).

### 12.3 Phase 2 — auxiliary transcription objective, for the universal-representation question

Rather than feeding notation at test time (the leakage trap above), use it as a **training
signal only**: train the model to produce a transcription *from* audio as an auxiliary
objective, then discard that output head at test time and re-run the existing battery + L2
probes on the resulting audio encoder. This can't leak the answer, because the transcription
is never present at inference — it tests whether the *pressure* to transcribe accurately
during training reshapes the encoder into something more decodable, not whether the model
can read an answer off a rendered score.

**Format decision, resolved 2026-08-12** (was parked, unresolved, in `RUPALI_READ_THIS.md` §5):
a compact JSON event list of audio-derived `(onset, dur, hz)` triples per detected note, e.g.
`[{"onset":0.23,"dur":0.95,"hz":261.6},...]`. Chosen over plain MIDI-as-text because pitch is
continuous Hz, not a quantized note number — satisfies requirement 5 (no continuous pitch was
the specific failure mode MIDI-as-text was ruled out for) while staying compact and
human-readable (requirements 2/4). Detection reuses the exact onset-detect + per-segment
piptrack + harmonic-rejection chain Track P (`render_piano_roll`) already verified for the
piano-roll image, just serialized as JSON text instead of plotted — implemented standalone in
`musicprobe/transcription_target.py` (duplicated, not imported, from
`scripts/render_harmony_repr.py`'s helpers, so this new consumer can never change Track P's
already-landed behavior). Built + run against the full 1248-stimulus battery 2026-08-12: 0
errors, 1104 unique audio_paths cached to `manifests/transcription_target.json` (median
target length 197 chars; a dense-rhythm tail runs up to ~4.1k chars — click-track-style
`tempo_bpm`/`beats_per_bar` stimuli generate many short notes — worth capping event count if
that proves too long a target once the multi-task trainer is built). This is a **training
target only**, generated from audio the same way Track P's image is, and never shown to the
model as input — the leakage risk that applies to every front-end track (D–Y) does not apply
here.

Unlike Track E's `f0_text` (pitch tasks only), this map is built over the **whole manifest**
by default (`build_map(tasks=None)`), since Phase 2 is meant to run across all three clusters
per PROJECT_STATE.md next action 17 — Rejected candidates from the original shortlist: prose
description (harder to make precisely comparable across seeds/checkpoints than a fixed
schema) and a pitch-bend-inclusive symbolic format (no existing audio-derived pitch-bend
estimator in this codebase to generate it from — would be new infrastructure, not a format
choice).

**Still open / not yet built**: the multi-task LoRA trainer itself (objective (a) unchanged +
objective (b) transcribe, per Design below) — CPU-side target generation is done, but nothing
GPU-side has been written or run. This is a bigger lift than Tracks L–Y's front-end injection
scripts (needs a multi-task loss/dataset-mixing design, not just a new prompt condition) —
see PROJECT_STATE.md next action 17 for scoping.

**Design**: multi-task LoRA fine-tune on Qwen2.5-Omni-7B — objective (a) unchanged, answer
the existing battery; objective (b) new, transcribe what's playing, in the chosen format.
Compare L2 probe accuracy (same battery of properties, same layers) before vs. after, on the
same leakage-safe splits as Track B. A win here is a stronger result than Phase 1's: it
would mean the *representation itself* got better, portably, not just that one more input
channel helped one model answer one battery.

### 12.4 What this is not

Video was considered and set aside for now: the current stimuli are synthesized from MIDI,
so there is no performance footage to film. It becomes relevant later specifically for the
deferred Carnatic module (§8), where a video of ornamented, continuous-pitch performance
(visible pitch bends, finger/vocal movement) could carry information audio alone doesn't —
not a fit for the current synthetic battery.

### 12.5 Sequencing

Phase 1 before Phase 2 — Phase 2 reuses Phase 1's fine-tuning stack, and Phase 1's result
tells us whether extra-modality-as-test-time-input is worth pursuing further before
investing in the harder auxiliary-objective design. Both phases target Qwen2.5-Omni-7B
first (the only open model with full Track B/C infrastructure that also accepts image
input); AF3/Music-Flamingo are excluded per §12.1 unless a future AF-Next release adds
vision support (check on release, same as the existing AF-Next action item in §3.5).

### 12.6 What actually happened (2026-07-26 to 07-30) — resolving decision 11's 5 requirements

Phase 1 ran, hit an OOD-training confound, and was superseded three times before landing on
a real answer (Tracks C–F, full numbers in PAPER.md). Net result against the 5 requirements
above, pitch only:

- **Requirement 1 (inferable from audio)**: yes for relative pitch (an external tracker
  reads it off the audio fine — Track E); absolute tuning needs a reference point alongside
  the value, which a tracker can supply (Track D-zoom) but which is not recoverable from
  audio alone via LoRA on the existing pathway (Track C's `llm_encoder` arm, encoder tuned,
  still null on both microtone tasks).
- **Requirement 2 (compact in tokens)**: text wins outright here — Track E's pitch-as-text
  front-end is a few tokens, no rendering pipeline, and fixes cents (0.62→0.92). The image
  route (Track D-zoom) needs a rendered chart with an explicit in-tune reference line to fix
  tuning (0.53→0.89) — bulkier, but it's the only method that closes that gap at all.
- **Requirement 3 (sufficient for the battery)**: confirmed only for the two microtone
  tasks so far (this was the shortlist Track B flagged as L2≫L3). `octave_id`/`note_count`
  turned out to be a *different* kind of gap — not perception-missing but
  readout-misaligned — and LoRA on the existing pathway alone fixes those (Track C,
  +0.50/+0.59 and +0.43/+0.40) with no new representation needed. So the ladder's answer is
  task-dependent: some gaps are alignment-fixable as-is, some need an explicit front-end.
- **Requirement 4 (human-readable)**: both surviving methods qualify — plain text
  ("current pitch: F#4 +12 cents") and a labeled chart are both legible to a person, unlike
  the rejected raw-embedding-fusion route (Track F): a trainable projector injecting a raw
  F0 feature into embedding space is demonstrably NOT human-readable and, separately, didn't
  work behaviorally even at 9x training data (Track F aug — the original run, `0aed136`, had
  a leakage bug in training-pitch sampling; fixed and rerun 2026-08-05, commits `8050378`/
  `b54c4bf`. Corrected audio-only-on-held-out: cents 0.62→0.74 clean (real but smaller gain
  than the leaky 0.89), tuning 0.51→0.62 clean, near the 2AFC majority rate — not evidence
  of learned absolute-tuning perception. Fusion-null verdict itself unaffected either way).
- **Requirement 5 (genre-universal, continuous pitch)**: both surviving front-ends
  (F0-as-text, F0-contour-plus-reference-line) are continuous-valued by construction — cents
  offsets and a contour line have no fixed-grid assumption, unlike staff notation/MIDI. This
  is the first concrete evidence the requirement is satisfiable for at least one property
  (pitch); genre-universality itself is still untested (all stimuli so far are Western
  12-TET-adjacent synths) and remains the Carnatic module's eventual stress test per §8.

**Practical recommendation for a model we fine-tune or design ourselves** (pitch only,
provisional pending non-pitch tasks and the deferred ladder arm, §6.6/next action #6 in
PROJECT_STATE.md): expose relative pitch via a compact in-context text feature (cheap, no
rendering); expose absolute tuning via a reference-anchored representation, not just the raw
value — a scale-anchored feature, not a sharper image, was the actual missing ingredient
(Track D force → zoom transition). Do not invest in learned raw-feature fusion into embedding
space at this project's data scale (~348 examples) — reuse a pretrained interface (numbers,
labeled charts) instead; Track F suggests this generalizes as a rule for any future
modality-injection experiment, not just pitch.

### 12.7 Tracks G/H (2026-08-05) — does the reference-line fix generalize?

Two follow-on tests of how far the D-zoom finding travels, run with the same paired
3-seed McNemar + wrong-condition mechanism-control discipline as C–F (no single-arm
Phase-1-style mistakes).

- **Track G — chromagram front-end for the harmonic cluster** (`key_id`/`mode_id`/
  `chord_quality`/`interval_id`, the one task group Tracks C–F never touched): a 12
  pitch-class × time chromagram is the harmonic analogue of the F0-contour that worked
  for pitch. **Clean null on all 4 tasks** — every 95% CI includes 0 (key_id Δ=−0.07,
  mode_id Δ=+0.04, chord_quality Δ=+0.13 n=72, interval_id Δ=+0.03; p≥.11 throughout).
  Mechanism controls: a wrong chromagram behaves like no chromagram on every task (the
  rare nudge isn't content-driven); wrong audio + correct chromagram only hurts `key_id`
  (p=.02) — the other three tasks show no evidence either way of leaning on audio vs image.
- **Track H — in-audio reference tone for `tuning_judgment`**: mixes a second tone into
  the clip instead of switching to a visual chart, testing whether D-zoom's fix is about
  "having a reference" in general or specifically about *reading an annotated position*.
  **Flat null** — reftone vs plain Δ=+0.01 (p=1.0); and the mechanism control is the
  decisive part: a *wrong* reftone (mistuned ±1–4 semitones) doesn't mislead the model any
  more than a correct one does (Δ=+0.02 vs plain, p=.82) — both ≈0.53, indistinguishable
  from no reference. The model isn't attempting a target-vs-reference comparison in audio
  at all.

**Updated reading**: the D-zoom fix is narrower than H7 originally framed it, on what's
been tested so far. It isn't "any same-level second modality" (ruled out by D conclusive/
force) and it isn't "any explicit reference, any channel" (ruled out by Track H, which
tested an audio reference tone specifically). Track G rules out one more thing — a flat,
unzoomed, unannotated chromagram — but **not** "any rendered chart in general": the zoom
and annotation ingredients that were both necessary for D-zoom (Track D force alone, zoom's
resolution-only precursor, didn't fix pitch either — see §12.6) were never combined and
tried on the harmonic cluster. So the honest current scope is: same-level-flat-image is
ruled out for both clusters; zoomed/annotated is confirmed necessary+sufficient for pitch
and simply untested for harmony. The harmonic cluster (`key_id`/`mode_id`/`chord_quality`/
`interval_id`) is the largest task group with zero *working* causal fix, but it also isn't
a clean "perception is entirely missing" case — the existing L2 own-encoder probe
(2026-07-24, already run, see PROJECT_STATE next action 13 for the numbers) finds modest
above-chance signal for 3 of the 4 tasks (`key_id` ~5x chance, `chord_quality`/`interval_id`
~2x chance) and only `mode_id` sits at essentially pure chance — and the L1 DSP floor
(2026-08-05, `l1_baselines.py`, pure signal processing, no learning at all) now agrees:
`mode_id` is the weakest of the six newly-covered tasks at only 25% (vs ~8% chance),
while `chord_quality`/`interval_id`/`key_id` all sit well above chance (54–80%). Three
independent methods (L1 DSP, L2 probe, L3 behavioral) now converge on `mode_id` being the
hardest task in the battery — worth weighting expectations accordingly across the sequence
below, and worth revisiting the known-gap note that mode melodies are random diatonic
walks rather than musician-composed (a stimulus-quality confound, not just a model one).

PROJECT_STATE next action 13 lays out the current follow-up plan, a fixed sequential
pipeline (Rupali's ordering, 2026-08-05, superseding the earlier I/J/K sketch): peak-picked
chroma → zoomed peak-picked chroma → line graph (multi-pitch trajectory, generalizing the
F0-contour that worked for pitch to polyphonic audio-derived content) → zoomed line graph →
piano-roll (Tracks L/M/N/O/P) — train and analyze each in order, stop at the first real
win. A text-based hint and an MCQ-template-image were both considered and explicitly ruled
out this round: the text hint would just repeat Track E's already-diagnosed
substitution-not-hearing pattern on a reference-giving idea already null'd once in-audio
(Track H); the MCQ-image assumes a question format the project doesn't want to depend on
long-term (the goal is open-ended, non-MCQ questions eventually).

**Two directions flagged as worth pursuing separately from the front-end sequence above**
(2026-08-05): (1) a **tonal-centroid / Tonnetz representation** — `librosa.feature.tonnetz`
(already in the venv) projects chroma onto a 6-D space (Harte, Sandler & Gasser 2006,
"Detecting Harmonic Change in Musical Audio") where harmonically-close relations (fifths,
thirds) map to small Euclidean distances; prototyped on a real `chord_quality` stimulus
2026-08-05 and confirmed it computes cleanly from audio (no leakage — same `chroma_cqt`
input as Track G, just a different linear projection). A genuinely different geometry from
anything tried so far, not just a resized/thresholded chromagram, though a quick n=3
sanity check on raw per-chord vectors didn't show obvious quality separation by eye — that's
expected from an unweighted single-frame average, not evidence against it; the real test is
whether training lets the model use the trajectory, same bar as everything else. (2) the
**auxiliary self-transcription training objective already specced in §12.3** but never run
(blocked on the transcription-format open question, `RUPALI_READ_THIS.md` §5) — this is the
one intervention in the whole project that asks the model to *generate* its own intermediate
representation during training rather than being handed one externally, which is the
long-run goal every front-end track here is really scaffolding toward. Resolving the format
question is the actual unblock, not more front-end iterations.

**§12.2 point 4 (own-encoder probe before/after) — resolved analytically, no GPU run
needed**: Tracks D/E/F's LoRA config (`train_track_d.py:build_lora_config`, reused by
`train_track_d_force.py`/`train_track_e_f0text.py`/`train_track_f_pitchfuse.py`) targets
`target_modules` matched only against `thinker.<lm_path>` (the LLM decoder found by
`_find_submodule`) — the regex never matches anything under `audio_tower` or the vision
tower. Those towers therefore receive zero gradient updates in every Track D/E/F run; their
forward computation on a given input is bit-identical before and after fine-tuning. So the
audio encoder's own representation provably does not change — the entire effect in Tracks
D/E/F is on the LLM-decoder / read side, exactly consistent with the wrong-audio /
wrong-feature substitution-not-hearing mechanism check already reported (feature+wrong_audio
≈ feature+correct_audio). Re-running `extract_activations.py --own-encoder` against these
checkpoints would reproduce Track B's existing own-encoder numbers exactly and add no new
information — skip it.

### 12.8 Status update (2026-08-05, results landed 2026-08-06) — L-Q/R-W: clean null

The plan in §12.7 is now built: all 6 harmony representations (L-Q) and their rhythm
analogues (R-W, mapped onto `tempo_bpm`/`beats_per_bar` — see PROJECT_STATE.md next
action 14 for the mapping table) have working renderers, verified against the full
1248-stimulus battery, and job manifests with sane held-out splits. Policy changed from
the original stop-early framing to running the full sequence per cluster and comparing
(Rupali's call, 2026-08-05) — the research question now explicitly includes "which
representation creates the richest usable signal," not just "does anything work."

Two additions beyond the original §12.7 plan, both from the same conversation: a tonal-
centroid/Tonnetz representation (Track Q) — `librosa.feature.tonnetz`, a 6-D projection
where harmonically-close pitches map to nearby points (Harte, Sandler & Gasser 2006) —
and its rhythm analogue, a "rhythm necklace" (Track W, Toussaint's circular rhythm
geometry, onset patterns as polygons on a circle). Both are real, audio-derived, non-
leaking representations, not resized/thresholded variants of something already tried.

GPU training/eval is the remaining step (`experiments/scripts/17_run_tracks_lq.sh` /
`18_run_tracks_rw.sh`, full context in `experiments/scripts/RUNBOOK_tracks_lq_rw.md`).
Not run yet as of this update.

**Results (2026-08-06, commit `6eef62a`, full 3-seed sweep)**: a clean null, worse
than null on two tasks. `key_id` is significantly HURT by every one of the 6 harmony
representations (Δacc −0.075 to −0.117, tonnetz p=.016) despite a solid 0.60-0.62
audio-only baseline — the image pulls a working signal down, not up. `chord_quality`
trends positive on all 6 (+0.08 to +0.17, piano-roll best, p=.043) but is
underpowered at n=72. `mode_id`/`interval_id` null throughout. On rhythm, `tempo_bpm`
is significantly hurt by 4/6 representations (p=.002 to .039) and trending negative
on the remaining 2 — no representation improves it. `beats_per_bar` null (n=33).
Mechanism: `wrong_image ≈ no_image` on both clusters (the image is mostly ignored),
but `image + wrong_audio` still hurts `key_id` (p=.02) — not fully inert. Net: the
D-zoom trick does not transfer to harmony or rhythm as tested — see §12.9 for why
this doesn't settle the question yet.

### 12.9 Tracks X/Y (2026-08-06) — the zoom+reference combination, not two more guesses

Re-examining why D-zoom worked for pitch, but nothing in §12.8 worked for harmony or
rhythm, surfaces a structural gap rather than a settled negative. D-zoom was not "zoom
alone" — Track D force (a sharper/forced-attention image, same information, no
reference line) was a clean null, mechanism-verified: the model demonstrably used the
image and accuracy still didn't move. It was not "reference alone" either — Track H
(an in-audio reference tone, no visual zoom) was a flat null, and the wrong-reference
control showed the model wasn't even attempting a comparison. Only **zoom and an
explicit annotated reference position together** fixed cents and absolute tuning.

Tracks L-Q and R-W tested those two ingredients as *separate* items on a six-step
list — zoom shows up alone in M/O/U, reference/structural richness shows up alone in
P/V — mirroring D-force's and Track H's single-ingredient tests, never D-zoom's
combination. So §12.8's null answers "does resolution alone help" and "does giving
the model more structure alone help" for harmony/rhythm — not yet "does the actual
D-zoom recipe, combined, help."

Two new tracks fill that specific cell:
- **Track X — zoomed peak-picked chroma + estimated-tonic reference.** Base: Track
  M's finer-hop chroma. Added: the tonic pitch class, estimated from the stimulus's
  own chroma via Krumhansl-profile correlation (the same method as the L1 baseline's
  `key_estimate`, restricted to just the tonic — mode is a nuisance parameter here),
  highlighted as a shaded reference band + text label, the harmonic analogue of
  D-zoom's red "in tune" line.
- **Track Y — zoomed rhythm-roll.** Base: Track V's onset-vs-detected-pulse-grid
  chart (the grid already comes from a detected periodicity, not the ground-truth
  beats-per-bar count — same leakage rule as before), rendered at Track U's finer
  time resolution instead of the default hop.

Both estimates remain strictly audio-derived — no manifest answer column is read by
either renderer, same discipline as every track since Track G's leakage audit.
CPU-side groundwork done and verified 2026-08-06 (both renderers run 1248/1248, 0
errors; held-out splits confirmed to exactly match their parent ladder — Track X
287/612 = Track G/L-Q's split, Track Y 127/132 = Track R-W's split, 0 overlap).
GPU training/eval pending (`experiments/scripts/19_run_tracks_xy.sh`,
`experiments/scripts/RUNBOOK_tracks_xy.md`).

If X/Y come back positive where L-Q/R-W didn't, that confirms D-zoom's fix is a real
combination effect generalizable beyond pitch, not a pitch-specific accident. If X/Y
are *also* null, that's a stronger and more interesting negative than §12.8's: it
would mean the ingredient that rescued pitch depends on pitch being a single
continuous scalar mappable onto one chart position — a property key/chord/tempo/meter
(categorical or multi-note aggregate judgments) don't share — rather than on nobody
having combined the right two ingredients yet.

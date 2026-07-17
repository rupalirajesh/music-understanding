"""Question construction: paraphrase templates, MCQ options with *diagnostic*
distractors, and deterministic choice-order permutation.

Distractors are chosen so that wrong answers carry signal (§0.6 of the plan):
for key ID the distractors are circle-of-fifths neighbors + the relative key;
for pitch, the semitone neighbors; for tempo, the octave errors. A model that
guesses uniformly across these looks different from one that mis-hears in a
musically structured way — that difference is half of RQ3.
"""
import numpy as np

from .theory import (NOTE_NAMES, INTERVALS, MODE_SPOKEN, CHORD_SPOKEN,
                     METERS, fifths_neighbors)

LETTERS = ["A", "B", "C", "D"]

MCQ_SUFFIX = ("\nAnswer with the letter of the correct option only "
              "(A, B, C, or D). Do not explain.")

# ---------------------------------------------------------------- templates
# >=3 paraphrases per task, so we measure the skill, not one phrasing.
TEMPLATES = {
    "pitch_note_id": [
        "Listen to this note. What is its pitch class (note name, ignoring octave)?",
        "A single note is played. Which note is it (C, C#, D, ...)?",
        "Identify the musical note you hear, by letter name only.",
    ],
    "octave_id": [
        "Listen to this note. In which octave is it, in scientific pitch notation (middle C = C4)?",
        "A single note is played. What octave number does it fall in (C4 = middle C)?",
        "Identify the octave of the note you hear, where middle C is octave 4.",
    ],
    "interval_id": [
        "Two notes are played. What is the musical interval between them?",
        "Listen to the two pitches. Name the interval they form.",
        "What interval do you hear between the notes in this clip?",
    ],
    "cents_discrimination": [
        "Two tones are played one after the other. Is the second tone higher, lower, or the same pitch as the first?",
        "Compare the two tones you hear. Relative to the first, is the second one higher in pitch, lower, or identical?",
        "You will hear tone 1, a pause, then tone 2. Is tone 2 higher, lower, or the same as tone 1?",
    ],
    "tempo_bpm": [
        "What is the tempo of this clip, in beats per minute? Answer with a number only.",
        "Estimate the BPM (beats per minute) of this rhythm. Reply with just the number.",
        "How fast is this clip in beats per minute? Give only a numeric answer.",
    ],
    "beats_per_bar": [
        "Listen to the accent pattern of this rhythm. How many beats are there per bar (one accented beat starts each bar)?",
        "This rhythm repeats in cycles, each starting with a stronger beat. How many beats long is each cycle?",
        "Count the beats between the strong accents. How many beats per measure does this rhythm have?",
    ],
    "tuning_judgment": [
        "A single note is played. Is it in tune (matching standard concert pitch, like a piano note) or out of tune?",
        "Listen to this note. Would it match a key on a well-tuned piano, or is it off-pitch (between piano notes)?",
        "Is the note you hear in tune or out of tune relative to standard Western tuning?",
    ],
    "key_id": [
        "What key is this music in?",
        "Listen to this clip and identify its musical key (e.g. D major, F# minor).",
        "Name the key of the passage you just heard.",
    ],
    "mode_id": [
        "What scale or mode is this music based on?",
        "Listen to the clip. Which scale/mode is being used?",
        "Identify the mode or scale type of this passage.",
    ],
    "chord_quality": [
        "A chord is played. What is its quality (e.g. major, minor, diminished)?",
        "Identify the type of the chord you hear.",
        "Listen to this chord. What kind of chord is it?",
    ],
    "progression_id": [
        "Which chord progression does this clip follow?",
        "Listen to the harmony. What progression is being played?",
        "Identify the chord progression in this passage.",
    ],
    "note_count": [
        "How many distinct notes are sounding simultaneously in this clip?",
        "Count the number of different pitches played at the same time. How many are there?",
        "How many notes make up the sound you hear (played together at once)?",
    ],
}

_ALL_MODES = list(MODE_SPOKEN.values())
_ALL_CHORDS = list(CHORD_SPOKEN.values())
_ALL_INTERVALS = [v[1] for v in INTERVALS.values()]
_PROGRESSIONS = ["I-IV-V-I", "I-V-vi-IV", "ii-V-I", "12-bar-blues"]


def mcq_options(task: str, truth: str, factors: dict, rng: np.random.Generator):
    """Return 4 options (truth included), with diagnostic distractors."""
    if task == "pitch_note_id":
        pc = NOTE_NAMES.index(truth)
        opts = {truth, NOTE_NAMES[(pc + 1) % 12], NOTE_NAMES[(pc - 1) % 12]}
        while len(opts) < 4:
            opts.add(NOTE_NAMES[rng.integers(12)])
    elif task == "octave_id":
        o = int(truth)
        opts = {str(o), str(o - 1), str(o + 1)}
        opts.add(str(o + 2) if rng.random() < 0.5 else str(o - 2))
    elif task == "interval_id":
        semi = factors["semitones"]
        near = [INTERVALS[s][1] for s in (semi - 1, semi + 1) if s in INTERVALS]
        opts = {truth, *near}
        while len(opts) < 4:
            opts.add(_ALL_INTERVALS[rng.integers(len(_ALL_INTERVALS))])
    elif task == "cents_discrimination":
        return ["higher", "lower", "same"], None  # fixed 3AFC, no letters needed
    elif task == "tempo_bpm":
        bpm = float(truth)
        opts = {truth, str(round(bpm * 2, 1)), str(round(bpm / 2, 1)),
                str(round(bpm * (1.2 if rng.random() < 0.5 else 0.8), 1))}
    elif task == "beats_per_bar":
        others = [str(METERS[m][0]) for m in METERS if str(METERS[m][0]) != truth]
        rng.shuffle(others)
        opts = {truth, *others[:3]}
    elif task == "tuning_judgment":
        return ["in tune", "out of tune"], None  # fixed 2AFC
    elif task == "key_id":
        opts = {truth, *fifths_neighbors(factors["tonic_pc"], factors["mode"])}
    elif task == "mode_id":
        others = [m for m in _ALL_MODES if m != truth]
        rng.shuffle(others)
        opts = {truth, *others[:3]}
    elif task == "chord_quality":
        others = [c for c in _ALL_CHORDS if c != truth]
        rng.shuffle(others)
        opts = {truth, *others[:3]}
    elif task == "progression_id":
        opts = set(_PROGRESSIONS)
    elif task == "note_count":
        n = int(truth)
        cands = [str(m) for m in range(max(1, n - 2), n + 3) if str(m) != truth]
        rng.shuffle(cands)
        opts = {truth, *cands[:3]}
    else:
        raise ValueError(f"no MCQ builder for task {task}")

    opts = sorted(opts)[:4]
    if truth not in opts:
        opts[0] = truth
    rng.shuffle(opts)  # position permutation happens HERE, per item
    return opts, LETTERS[opts.index(truth)]


def build_prompt(task: str, paraphrase_idx: int, fmt: str,
                 options: list[str] | None) -> str:
    q = TEMPLATES[task][paraphrase_idx % len(TEMPLATES[task])]
    if fmt == "explain":
        # Not auto-scored. Stored verbatim for manual analysis: a correct
        # explanation with specific notes/timestamps is evidence of listening;
        # but treat as evidence, not proof — models can confabulate coherent
        # rationales around a guessed answer.
        return (q + "\nFirst give your answer. Then explain exactly what you "
                "heard that supports it: name the specific notes/chords/beats "
                "and their approximate timestamps in seconds.")
    if fmt == "open" or task == "tempo_bpm":
        if task == "cents_discrimination":
            return q + "\nAnswer with exactly one word: higher, lower, or same."
        if task == "tuning_judgment":
            return q + "\nAnswer with exactly: in tune, or out of tune."
        return q + "\nGive a short, direct answer."
    lines = [q] + [f"{LETTERS[i]}. {o}" for i, o in enumerate(options)]
    return "\n".join(lines) + MCQ_SUFFIX

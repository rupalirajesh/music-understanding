"""Music-theory ground truth: note names, intervals, scales/modes, chords, keys.

Everything downstream (generators, prompts, scoring) imports names from here so
ground-truth vocabulary is defined exactly once.
"""

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Spoken names for the GM programs in config.GM_PROGRAMS (instrument_id task).
INSTRUMENT_SPOKEN = {
    "piano": "piano",
    "violin": "violin",
    "flute": "flute",
    "synth_lead": "synth lead",
}

INTERVALS = {  # semitones -> (short, spoken name)
    1: ("m2", "minor second"),
    2: ("M2", "major second"),
    3: ("m3", "minor third"),
    4: ("M3", "major third"),
    5: ("P4", "perfect fourth"),
    6: ("TT", "tritone"),
    7: ("P5", "perfect fifth"),
    8: ("m6", "minor sixth"),
    9: ("M6", "major sixth"),
    10: ("m7", "minor seventh"),
    11: ("M7", "major seventh"),
    12: ("P8", "octave"),
}

MODES = {  # name -> semitone degrees from tonic
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
}

MODE_SPOKEN = {
    "major": "major", "natural_minor": "natural minor",
    "harmonic_minor": "harmonic minor", "melodic_minor": "melodic minor",
    "dorian": "Dorian", "phrygian": "Phrygian", "lydian": "Lydian",
    "mixolydian": "Mixolydian", "locrian": "Locrian",
    "major_pentatonic": "major pentatonic", "minor_pentatonic": "minor pentatonic",
    "blues": "blues scale", "whole_tone": "whole-tone",
}

CHORDS = {  # name -> semitones from root
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented": [0, 4, 8],
    "sus4": [0, 5, 7],
    "major7": [0, 4, 7, 11],
    "minor7": [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
}

CHORD_SPOKEN = {
    "major": "major", "minor": "minor", "diminished": "diminished",
    "augmented": "augmented", "sus4": "suspended fourth (sus4)",
    "major7": "major seventh", "minor7": "minor seventh",
    "dominant7": "dominant seventh",
}

# Progressions expressed as (scale degree root offset in semitones, chord type)
PROGRESSIONS = {
    "I-IV-V-I": [(0, "major"), (5, "major"), (7, "major"), (0, "major")],
    "I-V-vi-IV": [(0, "major"), (7, "major"), (9, "minor"), (5, "major")],
    "ii-V-I": [(2, "minor7"), (7, "dominant7"), (0, "major7"), (0, "major7")],
    "12-bar-blues": [(0, "dominant7")] * 4 + [(5, "dominant7")] * 2
                    + [(0, "dominant7")] * 2 + [(7, "dominant7"), (5, "dominant7")]
                    + [(0, "dominant7"), (7, "dominant7")],
    "i-iv-v-i": [(0, "minor"), (5, "minor"), (7, "minor"), (0, "minor")],
    "i-VI-III-VII": [(0, "minor"), (8, "major"), (3, "major"), (10, "major")],
}

METERS = {  # name -> (beats per bar, accent pattern per beat: 1=strong, 0=weak)
    "3/4": (3, [1, 0, 0]),
    "4/4": (4, [1, 0, 0.5, 0]),
    "5/4": (5, [1, 0, 0.5, 0, 0]),
    "6/8": (6, [1, 0, 0, 0.5, 0, 0]),
    "7/8": (7, [1, 0, 0.5, 0, 0.5, 0, 0]),
}


def midi_to_name(midi: int) -> str:
    """60 -> 'C4' (scientific pitch notation, C4 = middle C)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_pitch_class(midi: int) -> str:
    return NOTE_NAMES[midi % 12]


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def key_name(tonic_pc: int, mode: str) -> str:
    quality = "major" if mode == "major" else "minor"
    return f"{NOTE_NAMES[tonic_pc]} {quality}"


def fifths_neighbors(tonic_pc: int, mode: str) -> list[str]:
    """Diagnostic MCQ distractors: circle-of-fifths neighbors + relative key.

    Errors toward these are honest listening errors; uniform errors are guessing.
    """
    out = [key_name((tonic_pc + 7) % 12, mode), key_name((tonic_pc + 5) % 12, mode)]
    if mode == "major":
        out.append(key_name((tonic_pc + 9) % 12, "minor"))  # relative minor
    else:
        out.append(key_name((tonic_pc + 3) % 12, "major"))  # relative major
    return out

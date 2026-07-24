"""Track B step 1 (runs on the H100 box): play every stimulus through an
encoder/model and save hidden states per layer.

Copy experiments/ to the GPU box, then:

  pip install torch torchaudio transformers soundfile pandas pyarrow accelerate
  python gpu/extract_activations.py --model m-a-p/MERT-v1-330M --out acts/mert330
  python gpu/extract_activations.py --model openai/whisper-large-v3 --out acts/whisper
  python gpu/extract_activations.py --model laion/clap-htsat-unfused --out acts/clap

Those three runs ARE the encoder-family comparison (music-SSL vs ASR vs
contrastive): same stimuli, same probe suite (probe.py), three pretraining
objectives — the sharpest architectural evidence in the study, since it
directly answers "which encoder should our fine-tune be built on."

For re-probing key_id/mode_id/chord_quality/interval_id against each LALM's
OWN audio encoder (not vanilla Whisper/MERT/CLAP — see PROFESSOR_UPDATE.md
open question #2: Music Flamingo/AF3 use AF-Whisper, a continued-pretrained
variant, and Qwen-Omni uses its own AuT-derived tower), pass --own-encoder:

  python gpu/extract_activations.py --model Qwen/Qwen2.5-Omni-7B --own-encoder --out acts/qwen25omni_own
  python gpu/extract_activations.py --model Qwen/Qwen3-Omni-30B-A3B-Instruct --own-encoder --out acts/qwen3omni_own
  python gpu/extract_activations.py --model nvidia/audio-flamingo-3-hf --own-encoder --out acts/af3_own
  python gpu/extract_activations.py --model nvidia/music-flamingo-2601-hf --own-encoder --out acts/musicflamingo_own

UNVERIFIED on hardware: the audio-tower submodule path is a best guess (see
_find_audio_tower's docstring) — its fallback will find *something* whose
class name looks like an audio encoder even if the guessed path is wrong,
but double check the printed path against the real architecture on first run.

Output: one .npz per stimulus: arrays "layer_00".."layer_NN", each
(n_frames, dim) fp16, mean-pooled over time to (dim,) additionally stored as
"pooled_00".. — probes use pooled for clip-level targets (key, mode, tempo)
and framewise for time-local targets (ornaments; H5 temporal-resolution test).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch


def load_mert(name: str):
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    model = AutoModel.from_pretrained(name, trust_remote_code=True,
                                      output_hidden_states=True).eval().cuda()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)

    def forward(wav: np.ndarray, sr: int):
        import torchaudio
        t = torch.tensor(wav, dtype=torch.float32)
        if sr != fe.sampling_rate:
            t = torchaudio.functional.resample(t, sr, fe.sampling_rate)
        inputs = fe(t.numpy(), sampling_rate=fe.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.cuda() for k, v in inputs.items()})
        return [h[0].cpu().to(torch.float16).numpy() for h in out.hidden_states]

    return forward


def load_whisper_encoder(name: str):
    """ASR-supervised reference encoder — what most LALMs hear through."""
    import torchaudio
    from transformers import WhisperModel, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name).encoder.eval().cuda()

    def forward(wav, sr):
        t = torch.tensor(wav, dtype=torch.float32)
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        feats = proc(t.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            # whisper-large-v3 may load in fp16; match the feature dtype to it
            out = enc(feats.cuda().to(next(enc.parameters()).dtype),
                      output_hidden_states=True)
        return [h[0].cpu().to(torch.float16).numpy() for h in out.hidden_states]

    return forward


def load_clap(name: str):
    """Contrastive clip-level encoder — expected strong on genre, weak on pitch."""
    import torchaudio
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).eval().cuda()

    def forward(wav, sr):
        t = torch.tensor(wav, dtype=torch.float32)
        if sr != 48000:
            t = torchaudio.functional.resample(t, sr, 48000)
        inputs = proc(audio=t.numpy(), sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            out = model.audio_model(**{k: v.cuda() for k, v in inputs.items()},
                                    output_hidden_states=True)
        return [h[0].reshape(-1, h.shape[-1]).cpu().to(torch.float16).numpy()
                for h in out.hidden_states]

    return forward


def _find_audio_tower(model, candidate_paths):
    """Walk a few likely attribute paths for the audio-encoder submodule
    (naming isn't standardized across model families / transformers versions);
    fall back to scanning named_modules() for a class name that looks like an
    audio encoder. UNVERIFIED — if this raises, run
    `for n, _ in model.named_modules(): print(n)` on the actual checkpoint and
    add the real path to candidate_paths above the fallback."""
    for path in candidate_paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj, path
        except AttributeError:
            continue
    hints = ("AudioTower", "AudioEncoder", "AuT", "WhisperEncoder")
    for name, mod in model.named_modules():
        if any(h in type(mod).__name__ for h in hints):
            return mod, name
    raise AttributeError(
        f"couldn't find an audio encoder submodule; tried {candidate_paths}. "
        "Run `for n,_ in model.named_modules(): print(n, type(_))` and add "
        "the real path to candidate_paths.")


def load_own_encoder_qwen_omni(model_name: str):
    """Extract Qwen2.5-Omni / Qwen3-Omni's OWN audio tower (not vanilla
    Whisper) — this is what the PROFESSOR_UPDATE open question #2 needs:
    re-probing key_id/mode_id/chord_quality/interval_id against the encoder
    each model actually hears through, not a generic standalone one.
    UNVERIFIED on hardware — the audio_tower path is a best guess from the
    Thinker-Talker architecture description; if _find_audio_tower's fallback
    has to kick in, note which path it found and hardcode it here."""
    import torchaudio
    import torch
    from transformers import AutoProcessor
    if "qwen3" in model_name.lower():
        from transformers import Qwen3OmniMoeForConditionalGeneration as Cls
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    model = Cls.from_pretrained(model_name, torch_dtype="auto",
                                device_map="cuda").eval()
    tower, found_path = _find_audio_tower(
        model, ["thinker.audio_tower", "audio_tower", "thinker.model.audio_tower"])
    print(f"[load_own_encoder_qwen_omni] using submodule at '{found_path}'")
    fe = processor.feature_extractor if hasattr(processor, "feature_extractor") \
        else processor.audio_processor
    target_sr = getattr(fe, "sampling_rate", 16000)
    # The Qwen-Omni audio tower forward is not a plain encoder: it needs
    # feature_lens + aftercnn_lens and the packed (masked) input_features, exactly
    # as the model's own forward prepares them (modeling_qwen2_5_omni ~L1801-1812).
    # Use the full processor to get the frame-level feature_attention_mask (the raw
    # feature_extractor's attention_mask is sample-level and wrong here).
    _conv = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                          {"type": "text", "text": "."}]}]
    _text = processor.apply_chat_template(_conv, add_generation_prompt=True,
                                          tokenize=False)
    dev = model.device
    tdtype = next(tower.parameters()).dtype

    def forward(wav: np.ndarray, sr: int):
        t = torch.tensor(wav, dtype=torch.float32)
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)
        pi = processor(text=_text, audio=[t.numpy()], sampling_rate=target_sr,
                       return_tensors="pt")
        feat = pi["input_features"].to(dev)
        fam = pi["feature_attention_mask"].to(dev)
        flens = fam.sum(-1)                                   # (B,) valid frames
        feat = feat.permute(0, 2, 1)[fam.bool()].permute(1, 0).to(tdtype)  # (mel, frames)
        tkw = dict(feature_lens=flens, output_hidden_states=True, return_dict=True)
        # 2.5-Omni's tower also needs aftercnn_lens; 3-Omni's does not (no such method)
        if hasattr(tower, "_get_feat_extract_output_lengths"):
            tkw["aftercnn_lens"] = tower._get_feat_extract_output_lengths(flens)[0]
        with torch.no_grad():
            out = tower(feat, **tkw)
        return [h.reshape(-1, h.shape[-1]).cpu().to(torch.float16).numpy()
                for h in out.hidden_states]

    return forward


def load_own_encoder_flamingo(model_name: str):
    """Extract Audio Flamingo 3 / Music Flamingo's OWN encoder (AF-Whisper —
    Whisper-large-v3 continued-pretrained on audio/music, NOT vanilla
    whisper-large-v3). Same purpose as load_own_encoder_qwen_omni above.
    UNVERIFIED — audio_tower path guessed from the transformers-native "-hf"
    port's likely naming convention (matches vision_tower convention used
    elsewhere in transformers multimodal models)."""
    import torchaudio
    import torch
    from transformers import AutoProcessor
    if "music-flamingo" in model_name.lower():
        from transformers import MusicFlamingoForConditionalGeneration as Cls
    else:
        from transformers import AudioFlamingo3ForConditionalGeneration as Cls

    processor = AutoProcessor.from_pretrained(model_name)
    model = Cls.from_pretrained(model_name, torch_dtype=torch.float32,
                                device_map="cuda").eval()
    tower, found_path = _find_audio_tower(model, ["audio_tower", "model.audio_tower"])
    print(f"[load_own_encoder_flamingo] using submodule at '{found_path}'")
    fe = processor.feature_extractor if hasattr(processor, "feature_extractor") \
        else processor.audio_processor
    target_sr = getattr(fe, "sampling_rate", 16000)
    # AudioFlamingo3Encoder.forward(input_features, input_features_mask=...): it
    # needs the frame-level mask (the model calls it as a kwarg, not attention_mask).
    tdtype = next(tower.parameters()).dtype

    def forward(wav: np.ndarray, sr: int):
        t = torch.tensor(wav, dtype=torch.float32)
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)
        feat = fe(t.numpy(), sampling_rate=target_sr, return_tensors="pt",
                  return_attention_mask=True)
        input_features = feat["input_features"].to("cuda", tdtype)
        mask = feat.get("attention_mask")
        if mask is None:
            mask = torch.ones(input_features.shape[0], input_features.shape[-1],
                              dtype=torch.long)
        mask = mask.to("cuda")
        with torch.no_grad():
            out = tower(input_features, input_features_mask=mask,
                        output_hidden_states=True, return_dict=True)
        return [h.reshape(-1, h.shape[-1]).cpu().to(torch.float16).numpy()
                for h in out.hidden_states]

    return forward


LOADERS = {
    "mert": load_mert,
    "whisper": load_whisper_encoder,
    "clap": load_clap,
    "qwen-omni-own": load_own_encoder_qwen_omni,
    "flamingo-own": load_own_encoder_flamingo,
}


def main(model_name: str, out_dir: str, manifest="manifests/stimuli.parquet",
         stimuli_root=".", own_encoder: bool = False):
    low = model_name.lower()
    if own_encoder:
        kind = ("qwen-omni-own" if "omni" in low else
                "flamingo-own" if "flamingo" in low else None)
        assert kind, f"--own-encoder not supported for {model_name} (only Qwen-Omni/Flamingo families)"
    else:
        kind = ("mert" if "mert" in low else
                "whisper" if "whisper" in low else
                "clap" if "clap" in low else None)
    assert kind, f"no loader for {model_name} yet — add one to LOADERS"
    forward = LOADERS[kind](model_name)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(manifest)
    for i, row in enumerate(man.itertuples()):
        dst = out / (row.stimulus_id.replace("/", "__") + ".npz")
        if dst.exists():
            continue
        wav, sr = sf.read(Path(stimuli_root) / row.audio_path)
        layers = forward(wav, sr)
        arrays = {}
        for li, h in enumerate(layers):
            arrays[f"pooled_{li:02d}"] = h.mean(axis=0)
            if li in (0, len(layers) // 2, len(layers) - 1):  # framewise: 3 layers only (disk)
                arrays[f"layer_{li:02d}"] = h
        np.savez_compressed(dst, **arrays)
        if i % 50 == 0:
            print(f"{i}/{len(man)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="m-a-p/MERT-v1-330M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="manifests/stimuli.parquet")
    ap.add_argument("--stimuli-root", default=".")
    ap.add_argument("--own-encoder", action="store_true",
                    help="extract this LALM's own audio tower (Qwen-Omni/Flamingo "
                         "families) instead of a generic standalone encoder")
    args = ap.parse_args()
    main(args.model, args.out, args.manifest, args.stimuli_root, args.own_encoder)

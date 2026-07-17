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
Full LALMs (Audio Flamingo 3 / Qwen2.5-Omni) need model-specific loading —
add a loader per model below; the loop and storage format stay identical.

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
            out = enc(feats.cuda(), output_hidden_states=True)
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
        inputs = proc(audios=t.numpy(), sampling_rate=48000, return_tensors="pt")
        with torch.no_grad():
            out = model.audio_model(**{k: v.cuda() for k, v in inputs.items()},
                                    output_hidden_states=True)
        return [h[0].reshape(-1, h.shape[-1]).cpu().to(torch.float16).numpy()
                for h in out.hidden_states]

    return forward


LOADERS = {"mert": load_mert, "whisper": load_whisper_encoder, "clap": load_clap}
# TODO(Phase 3): add "af3" (Audio Flamingo 3: tap AF-Whisper encoder layers,
# post-projector, and LLM decoder layers at audio token positions) and
# "qwen-omni" loaders. Same storage format.


def main(model_name: str, out_dir: str, manifest="manifests/stimuli.parquet",
         stimuli_root="."):
    low = model_name.lower()
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
    args = ap.parse_args()
    main(args.model, args.out, args.manifest, args.stimuli_root)

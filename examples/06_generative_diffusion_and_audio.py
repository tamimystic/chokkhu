"""Example 6: Generative Diffusion, Flow Matching, LoRA Adapters & Audio Processing."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.models.audio import AST, melspectrogram, mfcc, stft
from chokkhu.models.generative import DDPM, FlowMatching, LoRALinear


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 6: GENERATIVE DIFFUSION, FLOW & AUDIO TRANSFORMERS")
    print("=" * 70)

    # 1. Denoising Diffusion Probabilistic Model (DDPM)
    print("[1] Generative DDPM Noise Scheduling:")
    ddpm = DDPM(data_dim=16, timesteps=100, beta_start=1e-4, beta_end=0.02)
    x0 = np.random.randn(4, 16)
    t = np.array([10, 30, 50, 80])
    noise = np.random.randn(4, 16)
    xt, _ = ddpm.q_sample(x0, t, noise)
    print(f"  DDPM diffused latent shape at timestep t: {xt.shape}")

    # 2. Optimal Transport Flow Matching
    print("\n[2] Flow Matching Continuous Velocity Field:")
    fm = FlowMatching(input_dim=8)
    sample_gen = fm.sample(num_samples=2, steps=10, method="rk4")
    print(f"  Flow matching generated vector trajectory: {sample_gen.shape}")

    # 3. LoRA Parameter-Efficient Adapter
    print("\n[3] LoRA Low-Rank Adaptation Linear Layer:")
    lora = LoRALinear(in_features=64, out_features=64, r=8, lora_alpha=16.0)
    x_in = np.random.randn(2, 64).astype(np.float32)
    y_out = lora.forward(x_in)
    print(f"  LoRA adapted forward output shape: {y_out.shape} (rank r=8)")

    # 4. Audio Processing: STFT, Mel-Spectrogram, MFCC
    print("\n[4] Audio Signal Processing from Scratch:")
    sample_rate = 16000
    duration_secs = 1.0
    time_arr = np.linspace(0, duration_secs, int(sample_rate * duration_secs))
    audio_waveform = np.sin(2 * np.pi * 440.0 * time_arr) + 0.3 * np.sin(
        2 * np.pi * 880.0 * time_arr
    )

    spec = stft(audio_waveform, n_fft=512, hop_length=160)
    mel_spec = melspectrogram(
        audio_waveform, sr=sample_rate, n_fft=512, hop_length=160, n_mels=40
    )
    mfcc_feats = mfcc(audio_waveform, sr=sample_rate, n_mfcc=13)
    print(f"  STFT complex spectrogram shape: {spec.shape}")
    print(f"  Log Mel-filterbank energy shape: {mel_spec.shape}")
    print(f"  MFCC acoustic features shape   : {mfcc_feats.shape}")

    # 5. Audio Spectrogram Transformer (AST)
    print("\n[5] Conformer & Audio Spectrogram Transformer (AST):")
    ast = AST(
        num_classes=5,
        in_channels=1,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        patch_size=8,
        stride=8,
    )
    audio_batch = np.random.randn(2, 1, 32, 32)
    class_logits = ast(audio_batch)
    print(f"  AST audio classification logits shape: {class_logits.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import os
from pathlib import Path
from torchvision.models import resnet18

# ----------------------------
# Model Definition
# ----------------------------
class ResNetAudio(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# Preprocessing
# ----------------------------
def load_and_preprocess(path, target_sr=16000, seconds=5.0, n_mels=128):
    # Use soundfile directly to avoid torchaudio backend issues
    wav, sr = sf.read(str(path), dtype='float32')
    
    # Convert to torch tensor and ensure correct shape (channels, samples)
    wav = torch.from_numpy(wav)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(0)  # Add channel dimension
    else:
        wav = wav.T  # Transpose to (channels, samples)
    
    # Convert stereo to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)

    # Pad or trim to fixed length
    samples = int(seconds * target_sr)
    if wav.shape[1] > samples:
        wav = wav[:, :samples]
    elif wav.shape[1] < samples:
        pad = samples - wav.shape[1]
        wav = nn.functional.pad(wav, (0, pad))

    # Generate mel-spectrogram
    melspec = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=n_mels, f_min=20, f_max=target_sr//2, power=2.0
    )
    spec = melspec(wav)
    spec = T.AmplitudeToDB(stype="power")(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.unsqueeze(0)

# ----------------------------
# Inference
# ----------------------------
def predict(model_path, audio_path, device="cpu"):
    device = torch.device(device)
    model = ResNetAudio().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    spec = load_and_preprocess(audio_path).to(device)
    with torch.no_grad():
        logits = model(spec)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    classes = ["Real", "Fake"]
    pred_idx = probs.argmax()
    return {"class": classes[pred_idx], "probs": {c: float(p) for c, p in zip(classes, probs)}}

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    model_path = "../models/best.pt"

    root = tk.Tk()
    root.withdraw()  # Hide the main window
    audio_file = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
    )
    if not audio_file:
        audio_file = "./sample_test.wav"

    result = predict(model_path, audio_file, device="cuda" if torch.cuda.is_available() else "cpu")
    print("Prediction:", result["class"])
    print("Probabilities:", result["probs"])
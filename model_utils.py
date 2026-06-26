"""
Model utilities for Speech Recognition
Handles model loading, MFCC extraction, and prediction
"""

import os
import torch
import torch.nn as nn
import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 40
N_FFT = 400       # 25 ms at 16 kHz
HOP_LENGTH = 160  # 10 ms at 16 kHz
NUM_CLASSES = 30  # 28 chars + blank (0) + unused padding index

CHARACTERS = list("abcdefghijklmnopqrstuvwxyz '")
CHAR_TO_INDEX = {c: i + 1 for i, c in enumerate(CHARACTERS)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}


class SpeechRecognitionModel(nn.Module):
    """LSTM-based Speech Recognition Model"""

    def __init__(self, input_size=N_MFCC, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def normalize_mfcc(mfcc):
    """Per-utterance MFCC normalization."""
    return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)


def extract_mfcc(audio_path=None, audio_data=None, sr=None, normalize=True):
    """
    Extract MFCC features from audio file or audio data.

    Returns:
        MFCC features (40, time_frames)
    """
    if audio_path:
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    elif audio_data is not None:
        audio = audio_data
        if sr is None:
            sr = SAMPLE_RATE
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
    else:
        raise ValueError("Either audio_path or audio_data must be provided")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    if normalize:
        mfcc = normalize_mfcc(mfcc)
    return mfcc


def encode_text(text):
    return [CHAR_TO_INDEX[c] for c in text if c in CHAR_TO_INDEX]


def load_model(model_path="model/lstm_ctc_model.pth"):
    """Load the trained model."""
    model = SpeechRecognitionModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.eval()
        return model
    raise FileNotFoundError(
        f"Model not found at {model_path}. Please train the model first (python train.py)."
    )


def decode_prediction(output):
    """Decode model output using greedy CTC decoding."""
    predicted_indices = torch.argmax(output, dim=2)
    predicted_text = ""
    previous = None

    for idx in predicted_indices[0]:
        idx = idx.item()
        if idx != previous and idx in INDEX_TO_CHAR:
            predicted_text += INDEX_TO_CHAR[idx]
        previous = idx

    return predicted_text.strip()


def predict_from_audio(model, audio_path=None, audio_data=None, sr=None):
    """Extract MFCC, run model, decode text."""
    mfcc = extract_mfcc(audio_path=audio_path, audio_data=audio_data, sr=sr, normalize=True)
    mfcc_tensor = torch.tensor(mfcc.T).unsqueeze(0).float()

    with torch.no_grad():
        output = model(mfcc_tensor)

    return decode_prediction(output)

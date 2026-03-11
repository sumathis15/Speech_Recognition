"""
Model utilities for Speech Recognition
Handles model loading, MFCC extraction, and prediction
"""

import os
import torch
import torch.nn as nn
import librosa
import numpy as np


class SpeechRecognitionModel(nn.Module):
    """LSTM-based Speech Recognition Model"""
    
    def __init__(self, input_size=40, hidden_size=256, num_layers=2):
        super(SpeechRecognitionModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, 30)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# Character vocabulary (same as training)
CHARACTERS = list("abcdefghijklmnopqrstuvwxyz '")
CHAR_TO_INDEX = {c: i+1 for i, c in enumerate(CHARACTERS)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}


def extract_mfcc(audio_path=None, audio_data=None, sr=None):
    """
    Extract MFCC features from audio file or audio data
    
    Args:
        audio_path: Path to audio file (if provided)
        audio_data: Audio data array (if provided)
        sr: Sampling rate (if audio_data provided)
    
    Returns:
        MFCC features (40, time_frames)
    """
    if audio_path:
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
    elif audio_data is not None:
        audio = audio_data
        if sr is None:
            sr = 16000
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
    else:
        raise ValueError("Either audio_path or audio_data must be provided")
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )
    
    return mfcc


def load_model(model_path="model/lstm_ctc_model.pth"):
    """
    Load the trained model
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded model in evaluation mode
    """
    model = SpeechRecognitionModel()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model not found at {model_path}. Please ensure the model is trained and saved.")


def decode_prediction(output):
    """
    Decode model output using CTC decoding
    
    Args:
        output: Model output tensor (batch_size, time_steps, num_classes)
    
    Returns:
        Decoded text string
    """
    predicted_indices = torch.argmax(output, dim=2)
    
    predicted_text = ""
    previous = None
    
    for idx in predicted_indices[0]:
        idx = idx.item()
        
        # Skip repeated characters (CTC rule)
        if idx != previous:
            if idx in INDEX_TO_CHAR:
                predicted_text += INDEX_TO_CHAR[idx]
        
        previous = idx
    
    return predicted_text.strip()


def predict_from_audio(model, audio_path=None, audio_data=None, sr=None):
    """
    Complete pipeline: Extract MFCC → Predict → Decode
    
    Args:
        model: Loaded SpeechRecognitionModel
        audio_path: Path to audio file (if provided)
        audio_data: Audio data array (if provided)
        sr: Sampling rate (if audio_data provided)
    
    Returns:
        Predicted text string
    """
    # Extract MFCC features
    mfcc = extract_mfcc(audio_path=audio_path, audio_data=audio_data, sr=sr)
    
    # Transpose for LSTM input: (time_steps, features)
    mfcc_input = mfcc.T
    
    # Convert to tensor and add batch dimension
    mfcc_tensor = torch.tensor(mfcc_input).unsqueeze(0).float()
    
    # Run model
    with torch.no_grad():
        output = model(mfcc_tensor)
    
    # Decode prediction
    predicted_text = decode_prediction(output)
    
    return predicted_text

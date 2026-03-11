# Sentence-Level Speech Recognition using LSTM Networks

## 🎯 Project Overview

This project implements a **Sentence-Level Speech Recognition System** using LSTM (Long Short-Term Memory) neural networks for real-time video subtitle generation and live captioning.

### Business Use Cases
- **Intelligent Live Captioning System** for Online Video Streaming
- **Real-Time Video Subtitle Generation** Using LSTM-Based Speech Recognition

## 🛠️ Technologies Used

- **Python** - Programming language
- **PyTorch** - Deep learning framework
- **Librosa** - Audio processing and MFCC extraction
- **Streamlit** - Web application framework for prototype
- **NumPy, Pandas** - Data processing
- **Jupyter** - Interactive development and analysis

## 📊 Model Architecture

- **Type:** LSTM (Long Short-Term Memory) Network
- **Architecture:** Bidirectional, 2 layers, 256 hidden units
- **Input Features:** MFCC (Mel Frequency Cepstral Coefficients) - 40 features
- **Output:** Character-level predictions (28 character vocabulary: a-z, space, apostrophe)
- **Loss Function:** CTC (Connectionist Temporal Classification)
- **Training:** 15 epochs on 8000 samples from LibriSpeech dataset

## 📁 Project Structure

```
Speech_Recognition/
├── data/
│   ├── raw/
│   │   └── LibriSpeech/
│   │       └── train-clean-100/
│   └── processed/
├── model/
│   └── lstm_ctc_model.pth          # Trained model weights
├── notebooks/
│   └── speech_recognition.ipynb    # Training and analysis notebook
├── app.py                          # Streamlit web application
├── model_utils.py                  # Model utilities and inference functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd C:\Speech_Recognition
   ```

2. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

### Running the Streamlit Application

1. **Activate the virtual environment** (if not already activated):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

3. **The app will open in your browser** at `http://localhost:8501`

## 📖 Usage Guide

### Streamlit Application Features

1. **Audio Input:**
   - Upload audio files (WAV, MP3, FLAC, M4A, OGG)
   - Record audio directly in the browser

2. **Transcription:**
   - Click "Transcribe Audio" to generate subtitles
   - View the generated text/caption

3. **Evaluation Metrics:**
   - Enter ground truth text to calculate:
     - Word Error Rate (WER)
     - Character Error Rate (CER)
     - Sentence Accuracy

4. **Visualization:**
   - Audio waveform visualization
   - MFCC feature visualization

### Using the Model Programmatically

```python
from model_utils import load_model, predict_from_audio

# Load the trained model
model = load_model("model/lstm_ctc_model.pth")

# Predict from audio file
predicted_text = predict_from_audio(model, audio_path="path/to/audio.wav")

print(f"Transcribed text: {predicted_text}")
```

## 📊 Evaluation Metrics

The model is evaluated using:

- **Word Error Rate (WER):** Measures word-level accuracy
- **Character Error Rate (CER):** Measures character-level accuracy
- **Sentence Accuracy:** Exact match percentage

## 📚 Dataset

- **LibriSpeech** (train-clean-100 subset)
  - Standard ASR (Automatic Speech Recognition) dataset
  - Audio format: Mono, 16 kHz sampling rate, FLAC files
  - Contains read English speech from audiobooks

## 🔧 Model Training

The model training process is documented in `notebooks/speech_recognition.ipynb`:

1. Data loading and preprocessing
2. MFCC feature extraction
3. Model architecture definition
4. Training with CTC loss
5. Model evaluation and testing

## 📝 Project Deliverables

✅ Cleaned and preprocessed data  
✅ Deep learning model (LSTM with CTC)  
✅ Performance evaluation reports  
✅ Jupyter notebooks with documented analysis  
✅ Streamlit prototype for real-time captioning  

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure the trained model exists at `model/lstm_ctc_model.pth`
- If missing, train the model using the Jupyter notebook first

### Audio Loading Issues
- Ensure audio files are in supported formats (WAV, MP3, FLAC, etc.)
- Check that librosa can read the audio file

### Long Path Error (Windows)
- Enable Windows Long Path support (see project documentation)
- Or use shorter paths for the project directory

## 📄 License

This project is part of the GUVI HCL Skill Up program.

## 👥 Author

Developed as part of the Final Project 2 - 2026 requirements.

---

**Note:** This is a prototype for demonstration purposes. For production use, additional optimization and testing would be required.

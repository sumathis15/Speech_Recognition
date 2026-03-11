"""
Streamlit App for Real-Time Video Subtitle Generation
Intelligent Live Captioning System for Online Video Streaming
"""

import streamlit as st
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

from model_utils import load_model, predict_from_audio, extract_mfcc

# Page configuration
st.set_page_config(
    page_title="Speech Recognition - Live Captioning",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stAudio {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = ""
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'audio_sr' not in st.session_state:
    st.session_state.audio_sr = None


@st.cache_resource
def load_speech_model():
    """Load the speech recognition model (cached)"""
    try:
        model_path = "model/lstm_ctc_model.pth"
        model = load_model(model_path)
        return model, True
    except Exception as e:
        return None, str(e)


def plot_audio_waveform(audio_data, sr, title="Audio Waveform"):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 3))
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    ax.plot(time_axis, audio_data)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_mfcc_features(mfcc, title="MFCC Features"):
    """Plot MFCC features"""
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Coefficients")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Magnitude')
    plt.tight_layout()
    return fig


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎤 Speech Recognition System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Intelligent Live Captioning for Online Video Streaming</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load model
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                model, status = load_speech_model()
                if status is True:
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.error(f"❌ Error loading model: {status}")
                    st.stop()
        else:
            st.success("✅ Model loaded")
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("""
        **Architecture:** LSTM (Bidirectional, 2 layers)
        
        **Features:** MFCC (40 coefficients)
        
        **Training:** CTC Loss
        
        **Vocabulary:** 28 characters
        """)
        
       
    # Main content area
    tab1, tab2 = st.tabs(["🎵 Audio Transcription", "ℹ️ About"])
    
    with tab1:
        st.header("Audio Input")
        
        # Audio input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload Audio File", "Record Audio"],
            horizontal=True
        )
        
        audio_file = None
        audio_data = None
        sample_rate = None
        
        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload an audio file",
                type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Load audio
                try:
                    audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                    st.session_state.audio_data = audio_data
                    st.session_state.audio_sr = sample_rate
                    
                    # Display audio player
                    st.audio(tmp_path, format='audio/wav')
                    
                    # Show audio info
                    duration = len(audio_data) / sample_rate
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{duration:.2f} s")
                    with col2:
                        st.metric("Sample Rate", f"{sample_rate} Hz")
                    with col3:
                        st.metric("Channels", "Mono" if len(audio_data.shape) == 1 else "Stereo")
                    
                    audio_file = tmp_path
                    
                except Exception as e:
                    st.error(f"Error loading audio: {str(e)}")
                    audio_file = None
        
        else:  # Record Audio
            st.info("Click the microphone button below to record audio")
            audio_bytes = st.audio_input("Record audio", label_visibility="visible")
            
            if audio_bytes is not None:
                # Save recorded audio temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    # Handle both bytes and file-like objects
                    if hasattr(audio_bytes, 'read'):
                        tmp_file.write(audio_bytes.read())
                    else:
                        tmp_file.write(audio_bytes)
                    tmp_path = tmp_file.name
                
                # Load audio
                try:
                    audio_data, sample_rate = librosa.load(tmp_path, sr=None)
                    st.session_state.audio_data = audio_data
                    st.session_state.audio_sr = sample_rate
                    
                    # Display audio player
                    st.audio(tmp_path, format='audio/wav')
                    
                    # Show audio info
                    duration = len(audio_data) / sample_rate
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Duration", f"{duration:.2f} s")
                    with col2:
                        st.metric("Sample Rate", f"{sample_rate} Hz")
                    
                    audio_file = tmp_path
                    
                except Exception as e:
                    st.error(f"Error loading recorded audio: {str(e)}")
                    audio_file = None
        
        # Visualization
        if audio_data is not None:
            st.markdown("---")
            st.subheader("📈 Audio Visualization")
            
            st.pyplot(plot_audio_waveform(audio_data, sample_rate))
        
        # Transcription
        st.markdown("---")
        st.subheader("🎯 Transcription")
        
        if audio_file is not None or audio_data is not None:
            if st.button("🚀 Transcribe Audio", type="primary", use_container_width=True):
                with st.spinner("Processing audio and generating transcription..."):
                    try:
                        if audio_file:
                            predicted_text = predict_from_audio(
                                st.session_state.model,
                                audio_path=audio_file
                            )
                        else:
                            predicted_text = predict_from_audio(
                                st.session_state.model,
                                audio_data=audio_data,
                                sr=sample_rate
                            )
                        
                        st.session_state.prediction = predicted_text
                        st.success("✅ Transcription completed!")
                        
                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")
                        st.session_state.prediction = ""
            
            # Display prediction
            if st.session_state.prediction:
                st.markdown("### 📝 Generated Subtitle/Caption:")
                st.markdown(f'<div class="prediction-box"><p style="font-size: 1.3rem; line-height: 1.8;">{st.session_state.prediction}</p></div>', unsafe_allow_html=True)
                
                # Copy to clipboard button
                st.code(st.session_state.prediction, language=None)
        else:
            st.info("👆 Please upload or record an audio file to begin transcription")
    
    with tab2:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a **Sentence-Level Speech Recognition System** using LSTM (Long Short-Term Memory) 
        neural networks for real-time video subtitle generation and live captioning.
        
        ### 🏢 Business Use Cases
        
        - **Intelligent Live Captioning System** for Online Video Streaming
        - **Real-Time Video Subtitle Generation** Using LSTM-Based Speech Recognition
        
        ### 🛠️ Technical Details
        
        **Model Architecture:**
        - LSTM Network (Bidirectional, 2 layers, 256 hidden units)
        - Input: MFCC (Mel Frequency Cepstral Coefficients) - 40 features
        - Output: Character-level predictions (28 character vocabulary)
        - Loss Function: CTC (Connectionist Temporal Classification)
        
        **Data Preprocessing:**
        - Audio Format: Mono, 16 kHz sampling rate
        - MFCC Extraction: 40 coefficients
        - Frame length: 25 ms, Frame shift: 10 ms
        
        **Evaluation Metrics:**
        - Word Error Rate (WER)
        - Character Error Rate (CER)
        - Sentence Accuracy
        
        ### 📚 Dataset
        
        - **LibriSpeech** (train-clean-100 subset)
        - Standard ASR (Automatic Speech Recognition) dataset
        
        ### 🚀 Technologies Used
        
        - Python
        - PyTorch (Deep Learning Framework)
        - Librosa (Audio Processing)
        - Streamlit (Web Interface)
        - NumPy, Pandas (Data Processing)
        
        ### 📝 Project Deliverables
        
        ✅ Cleaned and preprocessed data  
        ✅ Deep learning model (LSTM with CTC)  
        ✅ Performance evaluation reports  
        ✅ Jupyter notebooks with documented analysis  
        ✅ Streamlit prototype for real-time captioning  
        
        ### 👨‍💻 Development
        
        This project follows best practices for:
        - Data handling and preprocessing
        - Model development and optimization
        - Code documentation and readability
        - Version control (Git)
        """)


if __name__ == "__main__":
    main()

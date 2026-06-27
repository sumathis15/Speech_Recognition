# Sentence-Level Speech Recognition using LSTM Networks

## Project Overview

This project implements a **Sentence-Level Speech Recognition System** using LSTM neural networks for live captioning / subtitle generation.

**Flow:** Audio in → MFCC features → LSTM + CTC → text out

## Technologies

- Python, PyTorch, Librosa, Streamlit, scikit-learn, jiwer

## Model Architecture

| Component | Detail |
|-----------|--------|
| Model | Bidirectional LSTM, 2 layers, 256 hidden units |
| Input | 40 MFCC features (16 kHz mono) |
| Output | Character-level (a–z, space, apostrophe) + CTC blank |
| Loss | CTC (Connectionist Temporal Classification) |
| Training | Full LibriSpeech `train-clean-100` (~28k clips) |

## Project Structure

```
Speech_Recognition/
├── app.py                          # Streamlit demo
├── train.py                        # Full-dataset training script
├── model_utils.py                  # Model, MFCC, predict
├── scripts/
│   └── export_test_upload_samples.py  # Export test-clean clips for app testing
├── notebooks/
│   ├── speech_recognition.ipynb    # Learning notebook
│   └── colab_train.ipynb           # Colab GPU training
├── data/
│   ├── README.md
│   ├── raw/LibriSpeech/            # Full dataset (gitignored)
│   └── test_upload/                # Held-out test clips (in Git)
│       ├── test_clean_01.flac
│       ├── test_clean_01.txt       # ground truth transcript
│       └── manifest.json
├── model/
│   └── lstm_ctc_model.pth          # Trained weights (gitignored — you add locally)
└── requirements.txt
```

## Getting Started

### 1. Install

```powershell
cd C:\Speech_Recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add the trained model

Place your trained weights at:

```
model/lstm_ctc_model.pth
```

(Train locally with `python train.py` or on Colab — see `notebooks/colab_train.ipynb`.)

### 3. Run the app

```powershell
streamlit run app.py
```

Open http://localhost:8501

---

## How to test fairly (important)

### Use `data/test_upload/` — not training clips, not random mic tests

| Audio source | Good for evaluation? | Why |
|--------------|---------------------|-----|
| **`data/test_upload/*.flac`** | **Yes — recommended** | Official LibriSpeech **test-clean** — never used in training |
| Clips from `train-clean-100` | Misleading | Model trained on these — looks better than it is |
| Your own voice / laptop mic | Often poor | Different mic, accent, casual speech vs read audiobooks |

The model was trained on **read English audiobook speech** (LibriSpeech). It is normal for your own voice to perform worse than test-clean clips.

### Test in Streamlit

1. Run `streamlit run app.py`
2. **Upload Audio File** → pick e.g. `data/test_upload/test_clean_01.flac`
3. Click **Transcribe Audio**
4. Open `data/test_upload/test_clean_01.txt` and compare to the prediction

All ground-truth transcripts are in `data/test_upload/manifest.json`.

### Regenerate test samples

Requires a one-time download of LibriSpeech **test-clean** (~346 MB):

```powershell
python scripts/export_test_upload_samples.py --download --num-samples 10
```

This creates 10 random clips from the official test set in `data/test_upload/`. Only this small folder is committed to Git — not the full dataset.

---

## Dataset splits (LibriSpeech)

| Split | Used for | In this project |
|-------|----------|-----------------|
| **train-clean-100** | Training | `train.py` / Colab training |
| **test-clean** | Final evaluation | `data/test_upload/` export script |
| Validation (10%) | During training | Random holdout inside `train.py` |

Download links (OpenSLR resource 12):

- Train: http://www.openslr.org/resources/12/train-clean-100.tar.gz (~6.3 GB)
- Test: http://www.openslr.org/resources/12/test-clean.tar.gz (~346 MB)

---

## Training

**Local:**

```powershell
python train.py --epochs 30 --batch-size 16
```

**Colab (GPU):** see `notebooks/colab_train.ipynb`

**Resume after disconnect:**

```powershell
python train.py --resume --start-epoch 22 --checkpoint-dir path/to/checkpoints ...
```

See `Speech Recognition Guide.md` for full details.

---

## Evaluation metrics

- **WER** — Word Error Rate (lower is better)
- **CER** — Character Error Rate (lower is better)

Reported during training on the validation split. Use `test_upload/` clips for manual demo testing.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Model not found | Add `model/lstm_ctc_model.pth` |
| Own voice works poorly | Expected — try `data/test_upload/` clips |
| Training clip works great | Expected — that data was in training set |
| Audio load error | Use WAV, MP3, FLAC, M4A, or OGG |

---

## Documentation

| File | Content |
|------|---------|
| `Speech Recognition Guide.md` | Full project learning guide |
| `Project Comparison Guide.md` | vs Fashion recommendation project |
| `data/test_upload/README.md` | Test sample usage |

---

## License / author

Part of the GUVI HCL Skill Up program — Final Project 2, 2026.

**Note:** Prototype for demonstration. Not production ASR quality.

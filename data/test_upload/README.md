# Test upload samples (LibriSpeech test-clean)

These audio files are from the **official LibriSpeech test-clean split** — not from the training set.

Use them to evaluate the Streamlit app fairly:

1. Run `streamlit run app.py`
2. Upload a file from this folder (e.g. `test_clean_01.flac`)
3. Click **Transcribe Audio**
4. Compare the prediction to the matching `.txt` file (ground truth)

## Files

| Audio | Ground truth |
|-------|----------------|
| `test_clean_01.flac` | `test_clean_01.txt` |
| ... | ... |
| `manifest.json` | All transcripts + LibriSpeech IDs |

## Why not use your own voice or training clips?

| Source | Expected result |
|--------|-----------------|
| **test_upload/** (test-clean) | Fair benchmark — model never trained on these |
| **train-clean-100 clip** | Often looks good — model may have memorized it |
| **Your own voice / mic** | Often worse — different mic, accent, noise, casual speech |

This LSTM was trained on read audiobook English (LibriSpeech). Casual speech and laptop microphones are harder.

## Regenerate samples

```bash
python scripts/export_test_upload_samples.py --download --num-samples 10
```

Requires ~346 MB one-time download of test-clean (stored under `data/raw/`, gitignored).

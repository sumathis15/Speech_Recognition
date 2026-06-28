# Data folders

| Folder | In Git? | Purpose |
|--------|---------|---------|
| `test_upload/` | **Yes** | Small held-out test clips for Streamlit app demos |
| `raw/` | No | Download only when training (not kept locally by default) |

## Test the app (recommended)

Use files in **`test_upload/`** — LibriSpeech **test-clean** clips the model never trained on.

```powershell
streamlit run app.py
# Upload: data/test_upload/test_clean_01.flac
# Compare to: data/test_upload/test_clean_01.txt
```

See `test_upload/README.md` and `manifest.json`.

## Training data (download when needed)

Not stored in this repo. Download on Colab or locally only if retraining:

| Split | URL | Size |
|-------|-----|------|
| train-clean-100 | http://www.openslr.org/resources/12/train-clean-100.tar.gz | ~6.3 GB |
| test-clean (for export script) | http://www.openslr.org/resources/12/test-clean.tar.gz | ~346 MB |

Extract to `data/raw/LibriSpeech/`.

Regenerate test upload samples:

```powershell
python scripts/export_test_upload_samples.py --download --num-samples 10
```

This writes to `test_upload/` only; you can delete `data/raw/` afterward to save disk space.

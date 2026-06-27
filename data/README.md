# Data folders

| Folder | In Git? | Purpose |
|--------|---------|---------|
| `raw/LibriSpeech/` | No (too large) | Full LibriSpeech download for training |
| `test_upload/` | **Yes** | Small held-out test clips for app demos |

## Training data

Download **train-clean-100** for training (~6.3 GB):

http://www.openslr.org/resources/12/train-clean-100.tar.gz

Extract to `data/raw/LibriSpeech/train-clean-100/`.

## Test data for the app

**Do not test with training clips** — the model has seen those during training.

Use **`test_upload/`** instead. These clips come from LibriSpeech **test-clean** (official held-out test set, never used in training).

To regenerate:

```bash
python scripts/export_test_upload_samples.py --download
```

See `test_upload/README.md` for how to use the samples in Streamlit.

"""
Export LibriSpeech test-clean clips for manual upload testing in the Streamlit app.

These samples come from the official TEST set (never used during training).

Usage:
    python scripts/export_test_upload_samples.py
    python scripts/export_test_upload_samples.py --download
    python scripts/export_test_upload_samples.py --num-samples 10
"""

import argparse
import json
import os
import random
import shutil
import tarfile
import urllib.request

DEFAULT_TEST_PATH = os.path.join("data", "raw", "LibriSpeech", "test-clean")
DEFAULT_OUTPUT = os.path.join("data", "test_upload")
TEST_ARCHIVE_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
TEST_ARCHIVE = os.path.join("data", "raw", "test-clean.tar.gz")


def load_librispeech_records(dataset_path):
    records = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.endswith(".trans.txt"):
                continue
            transcript_path = os.path.join(root, file)
            with open(transcript_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.strip().split(" ")
                    audio_id = parts[0]
                    text = " ".join(parts[1:]).lower()
                    audio_path = os.path.join(root, audio_id + ".flac")
                    if os.path.exists(audio_path):
                        records.append({"audio_path": audio_path, "text": text, "id": audio_id})
    return records


def download_test_clean():
    os.makedirs(os.path.dirname(TEST_ARCHIVE), exist_ok=True)
    if os.path.isdir(DEFAULT_TEST_PATH):
        flac_count = sum(
            1 for r, _, files in os.walk(DEFAULT_TEST_PATH) for f in files if f.endswith(".flac")
        )
        if flac_count > 0:
            print(f"test-clean already present ({flac_count} FLAC files).")
            return

    print(f"Downloading test-clean (~346 MB) from OpenSLR...")
    urllib.request.urlretrieve(TEST_ARCHIVE_URL, TEST_ARCHIVE)
    if not tarfile.is_tarfile(TEST_ARCHIVE):
        raise RuntimeError("Download failed — file is not a valid tar.gz archive.")

    print("Extracting test-clean...")
    with tarfile.open(TEST_ARCHIVE, "r:gz") as archive:
        archive.extractall(path=os.path.join("data", "raw"))
    os.remove(TEST_ARCHIVE)
    print("test-clean ready.")


def export_samples(test_path, output_dir, num_samples, seed):
    records = load_librispeech_records(test_path)
    if not records:
        raise FileNotFoundError(
            f"No records found in {test_path}. Run with --download or extract test-clean manually."
        )

    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("test_clean_") and (name.endswith(".flac") or name.endswith(".txt")):
                os.remove(os.path.join(output_dir, name))
    else:
        os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    chosen = random.sample(records, min(num_samples, len(records)))
    manifest = []

    for index, row in enumerate(chosen, start=1):
        base = f"test_clean_{index:02d}"
        flac_name = f"{base}.flac"
        txt_name = f"{base}.txt"
        shutil.copy2(row["audio_path"], os.path.join(output_dir, flac_name))
        with open(os.path.join(output_dir, txt_name), "w", encoding="utf-8") as handle:
            handle.write(row["text"] + "\n")
        manifest.append(
            {
                "file": flac_name,
                "transcript_file": txt_name,
                "librispeech_id": row["id"],
                "transcript": row["text"],
            }
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Exported {len(manifest)} test-clean samples to {output_dir}/")
    print(f"Manifest: {manifest_path}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Export LibriSpeech test-clean upload samples")
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download test-clean from OpenSLR if not present (~346 MB)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.download:
        download_test_clean()
    export_samples(args.test_path, args.output, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

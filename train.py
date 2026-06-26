"""
Train LSTM-CTC speech recognition model on full LibriSpeech train-clean-100.

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 16
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from jiwer import cer, wer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_utils import (
    CHAR_TO_INDEX,
    SpeechRecognitionModel,
    decode_prediction,
    encode_text,
    extract_mfcc,
)

DEFAULT_DATA_PATH = os.path.join("data", "raw", "LibriSpeech", "train-clean-100")
DEFAULT_MODEL_PATH = os.path.join("model", "lstm_ctc_model.pth")
DEFAULT_BEST_MODEL_PATH = os.path.join("model", "lstm_ctc_model_best.pth")


def load_librispeech_dataset(dataset_path):
    """Load audio paths and transcripts from LibriSpeech."""
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
                    audio_file = os.path.join(root, audio_id + ".flac")
                    if os.path.exists(audio_file):
                        records.append({"audio_path": audio_file, "text": text})
    return records


class LibriSpeechDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        mfcc = extract_mfcc(audio_path=row["audio_path"], normalize=True)
        mfcc = torch.tensor(mfcc.T).float()
        target = torch.tensor(encode_text(row["text"]), dtype=torch.long)
        return mfcc, target, row["text"]


def collate_batch(batch):
    mfccs, targets, texts = zip(*batch)
    input_lengths = torch.tensor([m.size(0) for m in mfccs], dtype=torch.long)
    mfccs_padded = pad_sequence(mfccs, batch_first=True)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets)
    return mfccs_padded, input_lengths, targets_concat, target_lengths, texts


def evaluate(model, loader, device, ctc_loss):
    model.eval()
    total_loss = 0.0
    predictions = []
    references = []

    with torch.no_grad():
        for mfccs, input_lengths, targets, target_lengths, texts in loader:
            mfccs = mfccs.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            outputs = model(mfccs)
            log_probs = outputs.log_softmax(2).transpose(0, 1)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            for i in range(mfccs.size(0)):
                actual_len = input_lengths[i].item()
                sample_out = outputs[i : i + 1, :actual_len, :]
                predictions.append(decode_prediction(sample_out))
                references.append(texts[i])

    avg_loss = total_loss / max(len(loader), 1)
    avg_wer = wer(references, predictions)
    avg_cer = cer(references, predictions)
    return avg_loss, avg_wer, avg_cer


def train_epoch(model, loader, optimizer, ctc_loss, device, grad_clip):
    model.train()
    total_loss = 0.0

    for mfccs, input_lengths, targets, target_lengths, _ in tqdm(loader, desc="Train", leave=False):
        mfccs = mfccs.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(mfccs)
        log_probs = outputs.log_softmax(2).transpose(0, 1)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM-CTC speech recognition model")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH, help="LibriSpeech train-clean-100 path")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Output model path")
    parser.add_argument("--best-model-path", default=DEFAULT_BEST_MODEL_PATH, help="Best checkpoint path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.data_path}. "
            "Download LibriSpeech train-clean-100 into data/raw/LibriSpeech/."
        )

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading full LibriSpeech dataset index...")
    records = load_librispeech_dataset(args.data_path)
    print(f"Full dataset: {len(records)} samples (no subsampling)")

    train_records, val_records = train_test_split(
        records, test_size=args.val_split, random_state=args.seed
    )
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    train_loader = DataLoader(
        LibriSpeechDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        LibriSpeechDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model = SpeechRecognitionModel().to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_cer = float("inf")
    print(f"\nStarting training for {args.epochs} epochs (batch_size={args.batch_size})...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, ctc_loss, device, args.grad_clip
        )
        val_loss, val_wer, val_cer = evaluate(model, val_loader, device, ctc_loss)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_WER={val_wer:.4f} | val_CER={val_cer:.4f} | "
            f"lr={current_lr:.6f} | time={elapsed/60:.1f}m"
        )

        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), args.best_model_path)
            print(f"  -> Saved best model (CER={best_cer:.4f}) to {args.best_model_path}")

    torch.save(model.state_dict(), args.model_path)
    print(f"\nTraining complete.")
    print(f"Final model: {args.model_path}")
    print(f"Best model (lowest val CER={best_cer:.4f}): {args.best_model_path}")

    # Copy best weights to default inference path
    if os.path.exists(args.best_model_path):
        torch.save(torch.load(args.best_model_path, map_location="cpu", weights_only=True), args.model_path)
        print(f"Deployed best checkpoint to {args.model_path} for inference.")


if __name__ == "__main__":
    main()

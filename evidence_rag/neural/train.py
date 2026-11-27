"""Train the from-scratch neural family tagger.

Usage:
    python -m evidence_rag.neural.train [--epochs 20] [--corpus data/processed/corpus.jsonl]

Artifacts written to outputs/neural/:
    vocab.json        token -> id mapping used at inference
    config.json       model + training hyperparameters
    model.pt          best-validation checkpoint (state_dict)
    train_log.jsonl   per-epoch train/val loss and macro-F1
    metrics.json      final summary
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evidence_rag.neural.data import (
    FAMILIES,
    build_vocab,
    encode,
    load_corpus,
    make_arrays,
)
from evidence_rag.neural.model import FamilyTagger

OUT_DIR = Path("outputs/neural")
SEED = 13
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM = 64
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_LEN = 200
MAX_VOCAB = 15000
VAL_FRACTION = 0.15
THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> float:
    y_pred = (y_prob >= threshold).astype(np.float32)
    scores = []
    for i in range(y_true.shape[1]):
        tp = float(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = float(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = float(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate(model: FamilyTagger, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y)) * x.shape[0]
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_true.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    true = np.concatenate(all_true)
    return total_loss / len(loader.dataset), macro_f1(true, probs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the neural family tagger from scratch")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-vocab", type=int, default=MAX_VOCAB)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_corpus(args.corpus)
    n = len(records)
    order = list(range(n))
    random.Random(SEED).shuffle(order)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = set(order[:n_val])
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]

    vocab = build_vocab(train_records, max_vocab=args.max_vocab)
    X_train, Y_train = make_arrays(train_records, vocab, max_len=args.max_len)
    X_val, Y_val = make_arrays(val_records, vocab, max_len=args.max_len)

    # Inverse-frequency positive weights for the multi-label BCE loss.
    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, 20.0), dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = FamilyTagger(len(vocab), len(FAMILIES), EMBED_DIM, HIDDEN_DIM, DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    log_path = OUT_DIR / "train_log.jsonl"
    log_path.write_text("")
    best_f1, best_state = -1.0, None
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        train_loss = epoch_loss / seen
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "seconds": round(time.time() - started, 1),
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(
        {
            "state_dict": best_state,
            "vocab": vocab,
            "families": FAMILIES,
            "config": {
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "max_len": args.max_len,
                "num_labels": len(FAMILIES),
            },
        },
        OUT_DIR / "model.pt",
    )
    (OUT_DIR / "config.json").write_text(
        json.dumps(
            {
                "task": "multi-label algorithm-family tagging",
                "supervision": "weak labels from algorithm-name mentions in title+abstract",
                "architecture": "Embedding(128) -> masked mean pool -> Linear(128,64)+ReLU+Dropout(0.3) -> Linear(64,4)",
                "loss": "BCEWithLogitsLoss with per-label inverse-frequency pos_weight",
                "optimizer": f"AdamW lr={LR} weight_decay={WEIGHT_DECAY}",
                "epochs": args.epochs,
                "batch_size": BATCH_SIZE,
                "train_records": len(train_records),
                "val_records": len(val_records),
                "vocab_size": len(vocab),
                "seed": SEED,
                "device": device,
                "best_val_macro_f1": round(best_f1, 4),
            },
            indent=2,
        )
    )
    metrics = {
        "best_val_macro_f1": round(best_f1, 4),
        "epochs_trained": args.epochs,
        "train_seconds": round(time.time() - started, 1),
        "checkpoint": "outputs/neural/model.pt",
        "note": "Weakly supervised labels; macro-F1 is measured against keyword-mention labels, not human judgment.",
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nbest val macro-F1: {best_f1:.4f}  -> {OUT_DIR}/model.pt")


if __name__ == "__main__":
    main()

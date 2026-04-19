import csv
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


DATA_PATH = Path("spamraw.csv")
CHECKPOINT_PATH = Path("spam_lstm_checkpoint.pt")

SEED = 42
VAL_SPLIT = 0.2
VOCAB_SIZE = 6000
MAX_LEN = 40
BATCH_SIZE = 64
EMBED_DIM = 128
LSTM_HIDDEN_DIM = 128
FC_HIDDEN_DIM = 64
DROPOUT = 0.3
LEARNING_RATE = 1e-3
EPOCHS = 6

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_text(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def load_sms_dataset(csv_path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        required = {"type", "text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain 'type' and 'text' columns")

        for row in reader:
            label = (row.get("type") or "").strip().lower()
            text = (row.get("text") or "").strip()
            if not text or label not in {"ham", "spam"}:
                continue
            texts.append(text)
            labels.append(1 if label == "spam" else 0)

    if not texts:
        raise ValueError("No valid samples were loaded from spamraw.csv")

    return texts, labels


def build_vocab(texts: list[str], vocab_size: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(preprocess_text(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max(0, vocab_size - 2)):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = preprocess_text(text)
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    token_ids = token_ids[:max_len]
    if len(token_ids) < max_len:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_len - len(token_ids)))
    return token_ids


def stratified_split(
    texts: list[str],
    labels: list[int],
    val_ratio: float,
    seed: int,
) -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    spam_idx = [i for i, y in enumerate(labels) if y == 1]
    ham_idx = [i for i, y in enumerate(labels) if y == 0]

    rng = random.Random(seed)
    rng.shuffle(spam_idx)
    rng.shuffle(ham_idx)

    spam_val_count = max(1, int(len(spam_idx) * val_ratio))
    ham_val_count = max(1, int(len(ham_idx) * val_ratio))

    val_indices = set(spam_idx[:spam_val_count] + ham_idx[:ham_val_count])
    train_indices = [i for i in range(len(labels)) if i not in val_indices]
    val_indices = sorted(val_indices)

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return (train_texts, train_labels), (val_texts, val_labels)


class SMSDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        x = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class SpamLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        lstm_hidden_dim: int,
        fc_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(lstm_hidden_dim, fc_hidden_dim)
        self.fc_out = nn.Linear(fc_hidden_dim, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids shape: [batch_size, seq_len]
        embedded = self.embedding(input_ids)
        # embedded shape: [batch_size, seq_len, embed_dim]

        lstm_output, (hidden_state, _) = self.lstm(embedded)
        # lstm_output shape: [batch_size, seq_len, lstm_hidden_dim]
        # hidden_state shape: [num_layers=1, batch_size, lstm_hidden_dim]

        last_hidden = hidden_state[-1]
        # last_hidden shape: [batch_size, lstm_hidden_dim]

        hidden_features = torch.relu(self.fc_hidden(self.dropout(last_hidden)))
        # hidden_features shape: [batch_size, fc_hidden_dim]

        logits = self.fc_out(self.dropout(hidden_features)).squeeze(1)
        # logits shape: [batch_size]
        return logits


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_correct / max(1, total)
    return epoch_loss, epoch_acc


def save_checkpoint(
    path: Path,
    model: nn.Module,
    vocab: dict[str, int],
    max_len: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "max_len": max_len,
        "model_params": {
            "vocab_size": len(vocab),
            "embed_dim": EMBED_DIM,
            "lstm_hidden_dim": LSTM_HIDDEN_DIM,
            "fc_hidden_dim": FC_HIDDEN_DIM,
            "dropout": DROPOUT,
        },
        "threshold": 0.5,
        "label_map": {0: "Not Spam", 1: "Spam"},
    }
    torch.save(checkpoint, path)


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels = load_sms_dataset(DATA_PATH)
    (train_texts, train_labels), (val_texts, val_labels) = stratified_split(
        texts, labels, VAL_SPLIT, SEED
    )

    vocab = build_vocab(train_texts, VOCAB_SIZE)

    train_dataset = SMSDataset(train_texts, train_labels, vocab, MAX_LEN)
    val_dataset = SMSDataset(val_texts, val_labels, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SpamLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        fc_hidden_dim=FC_HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Loaded {len(texts)} samples | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Using device: {device}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    save_checkpoint(CHECKPOINT_PATH, model, vocab, MAX_LEN)
    print(f"Checkpoint saved to: {CHECKPOINT_PATH.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CTC training sketch for tajweed sifat + rule detection.

Manifest JSONL (one item per line):
{
  "audio": "path/to/audio.wav",
  "sura": 1,
  "ayah": 1,
  "text": "optional override text"
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import torchaudio
except ImportError:  # pragma: no cover - optional dependency
    torchaudio = None

try:
    from transformers import WhisperModel, WhisperProcessor
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None
    WhisperProcessor = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tajweed_embeddings.embedder import TajweedEmbedder  # noqa: E402


def _load_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TajweedCTCDataset(Dataset):
    def __init__(self, manifest_path: Path, embedder: TajweedEmbedder, sample_rate: int = 16000):
        self.items = _load_manifest(manifest_path)
        self.embedder = embedder
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def _load_audio(self, path: str) -> torch.Tensor:
        if torchaudio is None:
            raise RuntimeError("torchaudio is required to load audio.")
        wav, sr = torchaudio.load(path)
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        audio_path = item["audio"]
        sura = item.get("sura")
        ayah = item.get("ayah")
        text = item.get("text")

        if text is None:
            if sura is None or ayah is None:
                raise ValueError("Provide sura/ayah or explicit text in manifest.")
            text = self.embedder.get_text(sura, ayah, None)

        if sura is None or ayah is None:
            embeddings = self.embedder.text_to_embedding(1, 1, text)
        else:
            embeddings = self.embedder.text_to_embedding(sura, ayah, text if item.get("text") else None)

        n_letters = self.embedder.n_letters
        idx_rule_start = self.embedder.idx_rule_start
        n_rules = self.embedder.n_rules
        idx_sifat_start = self.embedder.idx_sifat_start
        n_sifat = self.embedder.n_sifat

        letter_ids: List[int] = []
        rule_targets: List[np.ndarray] = []
        sifat_targets: List[np.ndarray] = []
        for vec in embeddings:
            letter_ids.append(int(np.argmax(vec[:n_letters])))
            rule_targets.append(vec[idx_rule_start : idx_rule_start + n_rules])
            sifat_targets.append(vec[idx_sifat_start : idx_sifat_start + n_sifat])

        return {
            "audio": self._load_audio(audio_path),
            "letter_ids": torch.tensor(letter_ids, dtype=torch.long),
            "rules": torch.tensor(np.stack(rule_targets), dtype=torch.float),
            "sifat": torch.tensor(np.stack(sifat_targets), dtype=torch.float),
        }


def _pad_sequence(seqs: list[torch.Tensor], padding_value: float) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=padding_value)


def build_collate(processor, blank_id: int):
    def collate(batch: list[dict]) -> dict:
        audios = [b["audio"] for b in batch]
        letter_ids = [b["letter_ids"] for b in batch]
        rules = [b["rules"] for b in batch]
        sifat = [b["sifat"] for b in batch]

        features = processor(audios, sampling_rate=16000, return_tensors="pt").input_features
        labels = _pad_sequence(letter_ids, padding_value=blank_id)
        label_lengths = torch.tensor([len(x) for x in letter_ids], dtype=torch.long)
        rules_pad = _pad_sequence(rules, padding_value=0.0)
        sifat_pad = _pad_sequence(sifat, padding_value=0.0)
        mask = torch.arange(labels.size(1))[None, :] < label_lengths[:, None]

        return {
            "input_features": features,
            "labels": labels,
            "label_lengths": label_lengths,
            "rules": rules_pad,
            "sifat": sifat_pad,
            "mask": mask,
        }

    return collate


def ctc_forced_align(log_probs: torch.Tensor, targets: torch.Tensor, blank_id: int) -> list[Tuple[int, int]]:
    """
    Viterbi forced alignment for CTC.
    log_probs: (T, V), targets: (L,)
    Returns list of (start, end) frame spans per target label.
    """
    T, _ = log_probs.shape
    labels = targets.tolist()
    if not labels:
        return []
    ext = [blank_id]
    for lab in labels:
        ext.append(lab)
        ext.append(blank_id)
    S = len(ext)
    neg_inf = -1e9
    alpha = torch.full((T, S), neg_inf, device=log_probs.device)
    backp = torch.zeros((T, S), dtype=torch.long, device=log_probs.device)

    alpha[0, 0] = log_probs[0, blank_id]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
        backp[0, 1] = 0

    for t in range(1, T):
        for s in range(S):
            best_score = alpha[t - 1, s]
            best_state = s
            if s - 1 >= 0 and alpha[t - 1, s - 1] > best_score:
                best_score = alpha[t - 1, s - 1]
                best_state = s - 1
            if s - 2 >= 0 and ext[s] != ext[s - 2] and alpha[t - 1, s - 2] > best_score:
                best_score = alpha[t - 1, s - 2]
                best_state = s - 2
            alpha[t, s] = best_score + log_probs[t, ext[s]]
            backp[t, s] = best_state

    last_state = S - 1
    if S > 1 and alpha[T - 1, S - 2] > alpha[T - 1, S - 1]:
        last_state = S - 2

    states = [0] * T
    s = last_state
    for t in range(T - 1, -1, -1):
        states[t] = s
        s = int(backp[t, s])

    spans = []
    for i in range(len(labels)):
        state_idx = 2 * i + 1
        frames = [t for t, st in enumerate(states) if st == state_idx]
        if not frames:
            start = spans[-1][1] if spans else 0
            end = min(start + 1, T)
        else:
            start = frames[0]
            end = frames[-1] + 1
        spans.append((start, end))
    return spans


def pool_spans(frames: torch.Tensor, spans: list[Tuple[int, int]]) -> torch.Tensor:
    pooled = []
    for start, end in spans:
        if end <= start:
            end = min(start + 1, frames.size(0))
        pooled.append(frames[start:end].mean(dim=0))
    return torch.stack(pooled, dim=0)


class TajweedCTCModel(nn.Module):
    def __init__(self, model_name: str, n_letters: int, n_rules: int, n_sifat: int):
        super().__init__()
        if WhisperModel is None:
            raise RuntimeError("transformers is required to load Whisper.")
        self.encoder = WhisperModel.from_pretrained(model_name).encoder
        hidden = self.encoder.config.d_model
        self.ctc_head = nn.Linear(hidden, n_letters + 1)
        self.rules_head = nn.Linear(hidden, n_rules)
        self.sifat_head = nn.Linear(hidden, n_sifat)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(input_features).last_hidden_state
        return enc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-name", default="tarteel-ai/whisper-base-ar-quran")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    if WhisperProcessor is None:
        raise RuntimeError("transformers is required for WhisperProcessor.")

    embedder = TajweedEmbedder()
    blank_id = embedder.n_letters
    processor = WhisperProcessor.from_pretrained(args.model_name)

    dataset = TajweedCTCDataset(args.manifest, embedder)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate(processor, blank_id),
    )

    model = TajweedCTCModel(args.model_name, embedder.n_letters, embedder.n_rules, embedder.n_sifat)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    model.train()
    for _epoch in range(args.epochs):
        for batch in loader:
            enc = model(batch["input_features"])
            logits = model.ctc_head(enc)
            log_probs = F.log_softmax(logits, dim=-1)
            input_lengths = torch.full(
                (log_probs.size(0),),
                log_probs.size(1),
                dtype=torch.long,
            )
            ctc_loss = ctc_loss_fn(
                log_probs.transpose(0, 1),
                batch["labels"],
                input_lengths,
                batch["label_lengths"],
            )

            rules_losses = []
            sifat_losses = []
            for b in range(log_probs.size(0)):
                spans = ctc_forced_align(log_probs[b], batch["labels"][b][: batch["label_lengths"][b]], blank_id)
                pooled = pool_spans(enc[b], spans)
                rules_logits = model.rules_head(pooled)
                sifat_logits = model.sifat_head(pooled)
                mask = batch["mask"][b][: pooled.size(0)].float()

                rules_loss = F.binary_cross_entropy_with_logits(
                    rules_logits,
                    batch["rules"][b][: pooled.size(0)],
                    reduction="none",
                ).sum(dim=-1)
                rules_losses.append((rules_loss * mask).sum() / mask.sum())

                sifat_loss = F.binary_cross_entropy_with_logits(
                    sifat_logits,
                    batch["sifat"][b][: pooled.size(0)],
                    reduction="none",
                ).sum(dim=-1)
                sifat_losses.append((sifat_loss * mask).sum() / mask.sum())

            loss = ctc_loss + torch.stack(rules_losses).mean() + torch.stack(sifat_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"loss={loss.item():.4f} ctc={ctc_loss.item():.4f}")


if __name__ == "__main__":
    main()

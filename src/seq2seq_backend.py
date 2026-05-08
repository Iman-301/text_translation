from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch


class Seq2SeqBackendError(Exception):
    pass


# Target-language tags used in the source sentence.
# For reverse translation, we also allow <2en> (requires a bidirectional-trained checkpoint).
_MULTI_TAGS: dict[str, str] = {"fr": "<2fr>", "es": "<2es>", "de": "<2de>", "en": "<2en>", "am": "<2am>"}


@dataclass(frozen=True)
class Seq2SeqConfig:
    checkpoint_path: Path
    vocab_dir: Path
    target_lang: Literal["fr", "es", "de", "en", "am"] = "fr"
    method: Literal["greedy", "beam_search"] = "greedy"
    beam_width: int = 5
    max_length: int = 50
    device: Literal["cpu", "cuda"] = "cpu"
    embedding_dim: int = 256
    hidden_dim: int = 512
    cell_type: Literal["LSTM", "GRU"] = "LSTM"


def _repo_root() -> Path:
    # This file lives in <repo>/src/
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=2)
def _load_seq2seq_modules():
    """
    Load the "from scratch" seq2seq implementation living in <repo>/seq2seq/src/.
    We keep it isolated to avoid name collisions with <repo>/src/translator.py.
    """
    import importlib.util
    import sys

    root = _repo_root()
    seq2seq_src = root / "seq2seq" / "src"

    if not seq2seq_src.exists():
        raise Seq2SeqBackendError(
            f"Missing folder: {seq2seq_src}. Expected the from-scratch implementation under seq2seq/."
        )

    # Allow seq2seq internal imports like `from encoder import Encoder`.
    sys.path.insert(0, str(seq2seq_src))

    def load_module(unique_name: str, file_name: str):
        file_path = seq2seq_src / file_name
        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None or spec.loader is None:
            raise Seq2SeqBackendError(f"Could not import {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    encoder_m = load_module("_seq2seq_encoder", "encoder.py")
    decoder_m = load_module("_seq2seq_decoder", "decoder.py")
    attention_m = load_module("_seq2seq_attention", "attention.py")
    utils_m = load_module("_seq2seq_utils", "utils.py")
    translator_m = load_module("_seq2seq_translator", "translator.py")

    return encoder_m, decoder_m, attention_m, utils_m, translator_m


@lru_cache(maxsize=2)
def _load_seq2seq_runner(
    checkpoint_path: str,
    vocab_dir: str,
    device: str,
    embedding_dim: int,
    hidden_dim: int,
    cell_type: str,
):
    encoder_m, decoder_m, attention_m, utils_m, translator_m = _load_seq2seq_modules()

    ckpt = Path(checkpoint_path)
    vocab = Path(vocab_dir)

    if not vocab.exists():
        raise Seq2SeqBackendError(
            f"Vocab directory not found: {vocab}. "
            "Train/download data first to create data/vocab (see seq2seq/README.md)."
        )

    src_vocab_path = vocab / "src_vocab.pkl"
    tgt_vocab_path = vocab / "tgt_vocab.pkl"
    if not src_vocab_path.exists() or not tgt_vocab_path.exists():
        raise Seq2SeqBackendError(
            "Missing vocab files. Expected both:\n"
            f"- {src_vocab_path}\n"
            f"- {tgt_vocab_path}\n"
            "Create them by running the seq2seq data prep/training pipeline."
        )

    if not ckpt.exists():
        raise Seq2SeqBackendError(
            f"Checkpoint not found: {ckpt}. "
            "Train the from-scratch model to produce models/checkpoints/best_model.pt."
        )

    real_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    src_vocab = utils_m.load_vocab(str(src_vocab_path))
    tgt_vocab = utils_m.load_vocab(str(tgt_vocab_path))

    checkpoint = torch.load(str(ckpt), map_location=real_device)

    # Auto-infer model dimensions from checkpoint to avoid shape mismatches.
    # This keeps the UI/CLI simple: users don't need to remember training dims.
    enc_sd = checkpoint.get("encoder_state_dict", {})
    emb_w = enc_sd.get("embedding.weight")
    if emb_w is not None and hasattr(emb_w, "shape") and len(emb_w.shape) == 2:
        embedding_dim = int(emb_w.shape[1])

    hh_w = enc_sd.get("rnn.weight_hh_l0")
    if hh_w is not None and hasattr(hh_w, "shape") and len(hh_w.shape) == 2:
        # For both LSTM and GRU, the second dim corresponds to hidden_dim.
        hidden_dim = int(hh_w.shape[1])

    encoder = encoder_m.Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cell_type=cell_type,
    )

    attention = attention_m.BahdanauAttention(hidden_dim=hidden_dim)

    decoder = decoder_m.Decoder(
        vocab_size=len(tgt_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        attention=attention,
        cell_type=cell_type,
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    translator = translator_m.Translator(
        encoder=encoder,
        decoder=decoder,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=real_device,
    )

    return translator, real_device


def translate_with_seq2seq(sentence: str, config: Seq2SeqConfig) -> tuple[str, str]:
    if not sentence.strip():
        raise Seq2SeqBackendError("Input text is empty.")

    tag = _MULTI_TAGS.get(config.target_lang)
    if tag is None:
        raise Seq2SeqBackendError(
            f"Unknown target_lang='{config.target_lang}'. Supported: {', '.join(sorted(_MULTI_TAGS))}"
        )

    translator, real_device = _load_seq2seq_runner(
        checkpoint_path=str(config.checkpoint_path),
        vocab_dir=str(config.vocab_dir),
        device=config.device,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        cell_type=config.cell_type,
    )

    tagged_sentence = f"{tag} {sentence}".strip()
    translation, _attention = translator.translate(
        tagged_sentence,
        max_length=config.max_length,
        method=config.method,
        beam_width=config.beam_width,
    )

    # Show actual inferred dims (not the user's defaults).
    inferred_emb = getattr(translator.encoder.embedding.weight, "shape", [None, None])[1]
    inferred_hid = getattr(translator.encoder.rnn.weight_hh_l0, "shape", [None, None])[1]
    meta = (
        f"Scratch multilingual seq2seq (to={config.target_lang}, {config.cell_type}, {config.method}, device={real_device}, "
        f"emb={inferred_emb}, hid={inferred_hid})"
    )
    return translation, meta


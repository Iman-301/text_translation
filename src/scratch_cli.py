from __future__ import annotations

import argparse
from pathlib import Path

from seq2seq_backend import Seq2SeqBackendError, Seq2SeqConfig, translate_with_seq2seq


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Translate text using from-scratch Seq2Seq + Attention.")
    p.add_argument("--text", required=True, help="Source sentence to translate")
    p.add_argument(
        "--to",
        choices=["fr", "es", "de"],
        default="fr",
        help="Target language for the multilingual model (default: fr)",
    )
    p.add_argument(
        "--method",
        choices=["greedy", "beam_search"],
        default="greedy",
        help="Decoding method (default: greedy)",
    )
    p.add_argument("--beam_width", type=int, default=5, help="Beam width (beam_search only)")
    p.add_argument("--max_length", type=int, default=50, help="Max generation length")

    p.add_argument(
        "--checkpoint",
        default="seq2seq/models/checkpoints/best_model.pt",
        help="Path to checkpoint (default: seq2seq/models/checkpoints/best_model.pt)",
    )
    p.add_argument(
        "--vocab_dir",
        default="seq2seq/data/vocab",
        help="Path to vocab dir (default: seq2seq/data/vocab)",
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="cpu/cuda (default: cpu)")
    p.add_argument("--embedding_dim", type=int, default=256, help="Must match training")
    p.add_argument("--hidden_dim", type=int, default=512, help="Must match training")
    p.add_argument("--cell_type", choices=["LSTM", "GRU"], default="LSTM", help="Must match training")
    return p


def main() -> None:
    args = build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = Seq2SeqConfig(
        checkpoint_path=(repo_root / args.checkpoint).resolve()
        if not Path(args.checkpoint).is_absolute()
        else Path(args.checkpoint),
        vocab_dir=(repo_root / args.vocab_dir).resolve()
        if not Path(args.vocab_dir).is_absolute()
        else Path(args.vocab_dir),
        target_lang=args.to,
        method=args.method,
        beam_width=args.beam_width,
        max_length=args.max_length,
        device=args.device,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        cell_type=args.cell_type,
    )

    try:
        out, meta = translate_with_seq2seq(args.text, cfg)
        print(out)
        print(f"[{meta}]")
    except Seq2SeqBackendError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


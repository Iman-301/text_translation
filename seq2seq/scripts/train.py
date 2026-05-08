"""
Training Script for Seq2Seq Translation

This script orchestrates the complete training pipeline:
1. Load and preprocess data
2. Build vocabularies
3. Initialize model
4. Train with validation
5. Save checkpoints and visualizations

Usage:
    python scripts/train.py --embedding_dim 256 --hidden_dim 512 --epochs 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vocabulary import Vocabulary
from encoder import Encoder
from decoder import Decoder
from attention import BahdanauAttention
from dataset import TranslationDataset
from trainer import Trainer
from utils import plot_loss_curves, count_parameters, save_vocab
from subword_tokenizer import SentencePieceConfig, SubwordTokenizer, train_sentencepiece


def build_vocabularies(train_src_file: str, train_tgt_file: str, min_freq: int = 2, max_size: int = 10000):
    """
    Build source and target vocabularies from training data.
    
    Args:
        train_src_file: Path to training source file
        train_tgt_file: Path to training target file
        min_freq: Minimum word frequency
        max_size: Maximum vocabulary size
    
    Returns:
        src_vocab, tgt_vocab: Vocabulary instances
    """
    print("Building vocabularies...")
    
    # Load sentences
    def load_sentences(file_path):
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().lower().split()
                sentences.append(tokens)
        return sentences
    
    src_sentences = load_sentences(train_src_file)
    tgt_sentences = load_sentences(train_tgt_file)
    
    # Build vocabularies
    src_vocab = Vocabulary()
    src_vocab.build_vocab(src_sentences, min_freq=min_freq, max_size=max_size)
    
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(tgt_sentences, min_freq=min_freq, max_size=max_size)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    return src_vocab, tgt_vocab


def _maybe_train_sentencepiece(
    *,
    vocab_dir: Path,
    train_src_file: Path,
    train_tgt_file: Path,
    cfg: SentencePieceConfig,
    force: bool,
) -> Path:
    spm_model = vocab_dir / "spm.model"
    spm_vocab = vocab_dir / "spm.vocab"
    if spm_model.exists() and spm_vocab.exists() and not force:
        return spm_model

    print("\nTraining SentencePiece BPE...")
    vocab_dir.mkdir(parents=True, exist_ok=True)
    model_path, _ = train_sentencepiece(
        input_paths=[train_src_file, train_tgt_file],
        model_prefix=vocab_dir / "spm",
        cfg=cfg,
    )
    # train_sentencepiece writes spm.model/spm.vocab; normalize name to spm.model
    if model_path.name != "spm.model":
        (vocab_dir / "spm.model").write_bytes(model_path.read_bytes())
    print(f"SentencePiece model saved to: {spm_model}")
    return spm_model


def _encode_line_with_tag(tokenizer: SubwordTokenizer, line: str) -> str:
    line = (line or "").strip()
    if not line:
        return ""
    parts = line.split()
    first = parts[0] if parts else ""
    has_tag = first.startswith("<2") and first.endswith(">") and len(first) <= 8
    if has_tag:
        tag = first
        rest = " ".join(parts[1:]).strip()
        pieces = tokenizer.encode(rest)
        return " ".join([tag] + pieces).strip()
    return " ".join(tokenizer.encode(line)).strip()


def _bpe_encode_file(tokenizer: SubwordTokenizer, in_path: Path, out_path: Path, *, keep_tag: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            if keep_tag:
                fout.write(_encode_line_with_tag(tokenizer, raw) + "\n")
            else:
                fout.write(" ".join(tokenizer.encode(raw)) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train seq2seq translation model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed_multilingual',
                       help='Directory containing processed data (default: data/processed_multilingual)')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory to save vocabularies')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints',
                       help='Directory to save model checkpoints')

    # Tokenization (SentencePiece BPE)
    parser.add_argument(
        "--use_bpe",
        action="store_true",
        help="If set, train/use SentencePiece BPE and encode train/val files before training.",
    )
    parser.add_argument("--bpe_vocab_size", type=int, default=16000, help="SentencePiece vocab size (default: 16000)")
    parser.add_argument("--bpe_model_type", type=str, default="bpe", help="SentencePiece model type (default: bpe)")
    parser.add_argument(
        "--bpe_character_coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage (default: 1.0; good for non-Latin scripts)",
    )
    parser.add_argument(
        "--bpe_force_train",
        action="store_true",
        help="If set, retrain SentencePiece model even if vocab_dir/spm.model exists.",
    )
    
    # Model arguments
    # Defaults tuned for better quality on CPU (~1 hour for the course-scale dataset).
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    parser.add_argument('--cell_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                       help='RNN cell type (default: LSTM)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability (default: 0.1)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold (default: 1.0)')
    parser.add_argument('--early_stopping_patience', type=int, default=4,
                       help='Early stopping patience (default: 4)')

    # Output / plotting
    parser.add_argument(
        '--plot_losses',
        action='store_true',
        help='If set, save loss curves with matplotlib (may fail on locked-down systems)'
    )
    
    # Vocabulary arguments
    parser.add_argument('--min_freq', type=int, default=1,
                       help='Minimum word frequency (default: 1)')
    parser.add_argument('--max_vocab_size', type=int, default=50000,
                       help='Maximum vocabulary size (default: 50000)')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
    
    print("=" * 60)
    print("Seq2Seq Translation Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Cell type: {args.cell_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.epochs}")
    print("=" * 60)
    
    # Build vocabularies
    train_src_file = Path(args.data_dir) / 'train.src'
    train_tgt_file = Path(args.data_dir) / 'train.tgt'
    
    if not train_src_file.exists() or not train_tgt_file.exists():
        print(f"\nError: Training data not found in {args.data_dir}")
        print("Please run: python scripts/download_data.py")
        return
    
    vocab_dir = Path(args.vocab_dir)

    # Optional: SentencePiece BPE encode the dataset first.
    if args.use_bpe:
        spm_cfg = SentencePieceConfig(
            vocab_size=int(args.bpe_vocab_size),
            model_type=str(args.bpe_model_type),
            character_coverage=float(args.bpe_character_coverage),
        )
        spm_model = _maybe_train_sentencepiece(
            vocab_dir=vocab_dir,
            train_src_file=train_src_file,
            train_tgt_file=train_tgt_file,
            cfg=spm_cfg,
            force=bool(args.bpe_force_train),
        )
        tokenizer = SubwordTokenizer(spm_model)

        bpe_dir = Path(args.data_dir) / "_bpe"
        bpe_train_src = bpe_dir / "train.src"
        bpe_train_tgt = bpe_dir / "train.tgt"
        bpe_val_src = bpe_dir / "val.src"
        bpe_val_tgt = bpe_dir / "val.tgt"

        print("\nEncoding dataset with SentencePiece...")
        _bpe_encode_file(tokenizer, train_src_file, bpe_train_src, keep_tag=True)
        _bpe_encode_file(tokenizer, train_tgt_file, bpe_train_tgt, keep_tag=False)
        _bpe_encode_file(tokenizer, Path(args.data_dir) / "val.src", bpe_val_src, keep_tag=True)
        _bpe_encode_file(tokenizer, Path(args.data_dir) / "val.tgt", bpe_val_tgt, keep_tag=False)

        train_src_file = bpe_train_src
        train_tgt_file = bpe_train_tgt
        val_src_file = bpe_val_src
        val_tgt_file = bpe_val_tgt
    else:
        val_src_file = Path(args.data_dir) / "val.src"
        val_tgt_file = Path(args.data_dir) / "val.tgt"

    src_vocab, tgt_vocab = build_vocabularies(
        str(train_src_file),
        str(train_tgt_file),
        min_freq=args.min_freq,
        max_size=args.max_vocab_size,
    )
    
    # Save vocabularies
    save_vocab(src_vocab, str(vocab_dir / 'src_vocab.pkl'))
    save_vocab(tgt_vocab, str(vocab_dir / 'tgt_vocab.pkl'))
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TranslationDataset(
        str(train_src_file),
        str(train_tgt_file),
        src_vocab,
        tgt_vocab
    )
    
    val_dataset = TranslationDataset(
        str(val_src_file),
        str(val_tgt_file),
        src_vocab,
        tgt_vocab
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=TranslationDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TranslationDataset.collate_fn
    )
    
    # Initialize model
    print("\nInitializing model...")
    encoder = Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        cell_type=args.cell_type,
        dropout=args.dropout
    )
    
    attention = BahdanauAttention(hidden_dim=args.hidden_dim)
    
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention=attention,
        cell_type=args.cell_type,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = count_parameters(encoder) + count_parameters(decoder)
    print(f"Total parameters: {total_params:,}")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Initialize trainer
    trainer = Trainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        grad_clip=args.grad_clip
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    history = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping_patience
    )

    # Plot loss curves (optional; can fail if font cache is unwritable)
    if args.plot_losses:
        print("\nPlotting loss curves...")
        try:
            plot_loss_curves(
                history['train_losses'],
                history['val_losses'],
                'visualizations/loss_curves.png'
            )
            print("Loss curves saved to: visualizations/loss_curves.png")
        except Exception as exc:
            print(f"Warning: could not plot loss curves: {exc}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Total training time: {history['total_time'] / 60:.1f} minutes")
    print(f"\nModel saved to: {args.checkpoint_dir}/best_model.pt")
    print(f"Vocabularies saved to: {args.vocab_dir}/")
    if args.plot_losses:
        print("Loss curves saved to: visualizations/loss_curves.png")
    print("\nNext steps:")
    print("  1. Translate sentences: python scripts/translate.py")
    print("  2. Visualize attention: python scripts/visualize_samples.py")


if __name__ == '__main__':
    main()

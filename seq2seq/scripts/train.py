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


def main():
    parser = argparse.ArgumentParser(description="Train seq2seq translation model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory to save vocabularies')
    parser.add_argument('--checkpoint_dir', type=str, default='models/checkpoints',
                       help='Directory to save model checkpoints')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension (default: 512)')
    parser.add_argument('--cell_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                       help='RNN cell type (default: LSTM)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability (default: 0.0)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping threshold (default: 1.0)')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience (default: 5)')
    
    # Vocabulary arguments
    parser.add_argument('--min_freq', type=int, default=2,
                       help='Minimum word frequency (default: 2)')
    parser.add_argument('--max_vocab_size', type=int, default=10000,
                       help='Maximum vocabulary size (default: 10000)')
    
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
    
    src_vocab, tgt_vocab = build_vocabularies(
        str(train_src_file),
        str(train_tgt_file),
        min_freq=args.min_freq,
        max_size=args.max_vocab_size
    )
    
    # Save vocabularies
    vocab_dir = Path(args.vocab_dir)
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
    
    val_src_file = Path(args.data_dir) / 'val.src'
    val_tgt_file = Path(args.data_dir) / 'val.tgt'
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
    
    # Plot loss curves
    print("\nPlotting loss curves...")
    plot_loss_curves(
        history['train_losses'],
        history['val_losses'],
        'visualizations/loss_curves.png'
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Total training time: {history['total_time'] / 60:.1f} minutes")
    print(f"\nModel saved to: {args.checkpoint_dir}/best_model.pt")
    print(f"Vocabularies saved to: {args.vocab_dir}/")
    print(f"Loss curves saved to: visualizations/loss_curves.png")
    print("\nNext steps:")
    print("  1. Translate sentences: python scripts/translate.py")
    print("  2. Visualize attention: python scripts/visualize_samples.py")


if __name__ == '__main__':
    main()

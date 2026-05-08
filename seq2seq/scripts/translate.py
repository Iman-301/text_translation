"""
Translation Script for Seq2Seq Model

This script translates sentences using a trained seq2seq model.

Usage:
    python scripts/translate.py --sentence "Hello world" --checkpoint models/checkpoints/best_model.pt
    python scripts/translate.py --sentence "The cat is black" --method beam_search
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch

from encoder import Encoder
from decoder import Decoder
from attention import BahdanauAttention
from translator import Translator
from utils import load_vocab


def main():
    parser = argparse.ArgumentParser(description="Translate sentences using trained model")
    
    parser.add_argument('--sentence', type=str, required=True,
                       help='Source sentence to translate')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory containing vocabularies')
    parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'beam_search'],
                       help='Decoding method (default: greedy)')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search (default: 5)')
    parser.add_argument('--max_length', type=int, default=50,
                       help='Maximum translation length (default: 50)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')

    # Model shape (must match training)
    parser.add_argument('--embedding_dim', type=int, default=256,
                       help='Embedding dim used during training (default: 256)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dim used during training (default: 512)')
    parser.add_argument('--cell_type', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                       help='RNN cell type used during training (default: LSTM)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Seq2Seq Translation")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Decoding method: {args.method}")
    if args.method == 'beam_search':
        print(f"Beam width: {args.beam_width}")
    print("=" * 60)
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    vocab_dir = Path(args.vocab_dir)
    src_vocab = load_vocab(str(vocab_dir / 'src_vocab.pkl'))
    tgt_vocab = load_vocab(str(vocab_dir / 'tgt_vocab.pkl'))
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model configuration from checkpoint
    # Must match the configuration used during training.
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    cell_type = args.cell_type
    
    print(f"Model configuration:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Cell type: {cell_type}")
    
    # Initialize model
    print("\nInitializing model...")
    encoder = Encoder(
        vocab_size=len(src_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cell_type=cell_type
    )
    
    attention = BahdanauAttention(hidden_dim=hidden_dim)
    
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        attention=attention,
        cell_type=cell_type
    )
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    # Initialize translator
    translator = Translator(
        encoder=encoder,
        decoder=decoder,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device
    )
    
    # Translate
    print("\n" + "=" * 60)
    print("Translation")
    print("=" * 60)
    print(f"Source: {args.sentence}")
    
    translation, attention_weights = translator.translate(
        args.sentence,
        max_length=args.max_length,
        method=args.method,
        beam_width=args.beam_width if args.method == 'beam_search' else 5
    )
    
    print(f"Translation: {translation}")
    print(f"Attention shape: {attention_weights.shape}")
    print("=" * 60)
    
    # Show attention summary
    if attention_weights.size > 0:
        print("\nAttention Summary:")
        src_tokens = args.sentence.lower().split()
        tgt_tokens = translation.split()
        
        print(f"Source tokens: {src_tokens}")
        print(f"Target tokens: {tgt_tokens}")
        
        # Show which source word each target word attended to most
        print("\nPrimary attention for each target word:")
        for i, tgt_token in enumerate(tgt_tokens):
            if i < attention_weights.shape[0]:
                max_idx = attention_weights[i].argmax()
                if max_idx < len(src_tokens):
                    src_token = src_tokens[max_idx]
                    attention_val = attention_weights[i, max_idx]
                    print(f"  '{tgt_token}' ← '{src_token}' (attention: {attention_val:.3f})")


if __name__ == '__main__':
    main()

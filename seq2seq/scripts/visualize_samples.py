"""
Attention Visualization Script

This script generates attention heatmaps for sample translations,
showing which source words the model focuses on for each target word.

Usage:
    python scripts/visualize_samples.py --checkpoint models/checkpoints/best_model.pt
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
from visualizer import AttentionVisualizer
from subword_tokenizer import SubwordTokenizer
from utils import load_vocab


# Sample sentences for visualization
SAMPLE_SENTENCES = [
    "The cat is black",
    "I love you",
    "Good morning",
    "How are you",
    "The weather is nice today",
    "She is reading a book",
    "We are learning French",
    "This is a simple sentence",
    "Thank you very much",
    "Have a good day"
]


def main():
    parser = argparse.ArgumentParser(description="Generate attention visualizations")
    
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_dir', type=str, default='data/vocab',
                       help='Directory containing vocabularies')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'beam_search'],
                       help='Decoding method (default: greedy)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize (default: 10)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Attention Visualization")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Decoding method: {args.method}")
    print(f"Number of samples: {args.num_samples}")
    print("=" * 60)
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    vocab_dir = Path(args.vocab_dir)
    src_vocab = load_vocab(str(vocab_dir / 'src_vocab.pkl'))
    tgt_vocab = load_vocab(str(vocab_dir / 'tgt_vocab.pkl'))

    tokenizer = None
    spm_model = vocab_dir / "spm.model"
    if spm_model.exists():
        try:
            tokenizer = SubwordTokenizer(spm_model)
            print(f"Loaded SentencePiece model: {spm_model}")
        except Exception as exc:
            print(f"Warning: could not load SentencePiece model ({exc}); using whitespace tokenization.")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Infer model configuration - use training defaults
    embedding_dim = 256
    hidden_dim = 512
    cell_type = 'LSTM'
    
    print(f"Model: {cell_type}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
    
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
    
    # Initialize translator and visualizer
    translator = Translator(
        encoder=encoder,
        decoder=decoder,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        tokenizer=tokenizer,
    )
    
    visualizer = AttentionVisualizer()
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    attention_list = []
    
    for i, sentence in enumerate(SAMPLE_SENTENCES[:args.num_samples]):
        print(f"\n[{i+1}/{args.num_samples}] Translating: {sentence}")
        
        # Translate
        translation, attention_weights = translator.translate(
            sentence,
            max_length=50,
            method=args.method
        )
        
        print(f"  Translation: {translation}")
        
        # Prepare tokens for visualization
        src_tokens = ['<SOS>'] + sentence.lower().split() + ['<EOS>']
        tgt_tokens = translation.split()
        
        # Adjust attention matrix to include <SOS> and <EOS>
        # The attention_weights from translator already includes these
        
        # Create title
        title = f"'{sentence}' → '{translation}'"
        
        # Add to list for batch visualization
        attention_list.append((
            attention_weights,
            src_tokens,
            tgt_tokens,
            title
        ))
        
        # Also analyze attention pattern
        analysis = visualizer.analyze_attention_pattern(
            attention_weights,
            src_tokens,
            tgt_tokens
        )
        
        print(f"  Attention: {analysis['entropy_interpretation']}, "
              f"{analysis['alignment_pattern']}")
    
    # Generate all visualizations
    print("\n" + "=" * 60)
    print("Saving Visualizations")
    print("=" * 60)
    
    visualizer.plot_multiple_attentions(
        attention_list,
        str(output_dir),
        prefix="attention_sample"
    )
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Saved {len(attention_list)} visualizations to {output_dir}/")
    print("\nYou can now:")
    print(f"  1. View the attention heatmaps in {output_dir}/")
    print("  2. Analyze the attention patterns")
    print("  3. Compare with human translation strategies")
    print("\nThese visualizations show:")
    print("  - Which source words the model focuses on")
    print("  - Word alignment patterns between languages")
    print("  - Reordering strategies (monotonic vs. non-monotonic)")


if __name__ == '__main__':
    main()

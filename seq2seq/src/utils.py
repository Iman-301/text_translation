"""
Utility Functions for Seq2Seq Translation

This module provides helper functions for tokenization, masking, and visualization.
"""

import os
from pathlib import Path
from typing import List

import torch


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization.
    
    For educational purposes, we use basic tokenization. Production systems
    would use more sophisticated methods (BPE, WordPiece, SentencePiece).
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens
    
    Example:
        >>> tokenize("Hello, world!")
        ['hello,', 'world!']
    """
    return text.lower().strip().split()


def create_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Create mask for padded positions.
    
    Args:
        lengths: [batch_size] - actual lengths of sequences
        max_length: Maximum length in the batch
    
    Returns:
        mask: [batch_size, max_length] - 1 for real tokens, 0 for padding
    """
    batch_size = lengths.size(0)
    mask = torch.arange(max_length, device=lengths.device).expand(batch_size, max_length)
    mask = (mask < lengths.unsqueeze(1)).long()
    return mask


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    # Avoid matplotlib/font cache issues in locked-down environments by forcing
    # its config/cache into the project directory.
    mpl_dir = Path(".mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curves saved to {save_path}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_vocab(vocab, path: str) -> None:
    """
    Save vocabulary to file.
    
    Args:
        vocab: Vocabulary instance
        path: Path to save vocabulary
    """
    import pickle
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {path}")


def load_vocab(path: str):
    """
    Load vocabulary from file.
    
    Args:
        path: Path to vocabulary file
    
    Returns:
        Vocabulary instance
    """
    import pickle
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {path}")
    return vocab

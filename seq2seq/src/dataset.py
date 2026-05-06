"""
Dataset and Data Loading for Neural Machine Translation

This module handles loading, preprocessing, and batching parallel corpus data
for training the seq2seq translation model.

Cognitive Science Connection:
The data preprocessing pipeline mirrors aspects of language acquisition:
- Tokenization: Segmenting continuous speech/text into words
- Vocabulary building: Learning which words exist in the language
- Sequence padding: Handling variable-length utterances
- Batching: Processing multiple examples simultaneously (like classroom learning)
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from pathlib import Path


class TranslationDataset(Dataset):
    """
    Dataset for parallel sentence pairs (source-target).
    
    This dataset handles the loading and preprocessing of aligned sentence pairs,
    similar to how language learners are exposed to parallel examples in both
    their native language and the target language.
    
    Args:
        src_file: Path to source language file (one sentence per line)
        tgt_file: Path to target language file (aligned with source)
        src_vocab: Source vocabulary instance
        tgt_vocab: Target vocabulary instance
        max_length: Maximum sentence length (longer sentences are truncated)
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_vocab,  # Vocabulary instance
        tgt_vocab,  # Vocabulary instance
        max_length: int = 50
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        # Load sentence pairs
        self.src_sentences = self._load_sentences(src_file)
        self.tgt_sentences = self._load_sentences(tgt_file)
        
        # Verify alignment
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            f"Source and target files must have same number of lines: " \
            f"{len(self.src_sentences)} vs {len(self.tgt_sentences)}"
    
    def _load_sentences(self, file_path: str) -> List[List[str]]:
        """
        Load and tokenize sentences from file.
        
        Simple whitespace tokenization is used for educational clarity.
        Production systems would use more sophisticated tokenizers (BPE, WordPiece).
        
        Args:
            file_path: Path to text file (one sentence per line)
        
        Returns:
            List of tokenized sentences (list of word lists)
        """
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Simple whitespace tokenization
                    # This mirrors basic word segmentation in language processing
                    tokens = line.lower().split()
                    
                    # Truncate if too long
                    if len(tokens) > self.max_length:
                        tokens = tokens[:self.max_length]
                    
                    sentences.append(tokens)
        return sentences
    
    def __len__(self) -> int:
        """Return number of sentence pairs."""
        return len(self.src_sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sentence pair as integer sequences.
        
        This retrieval process is like accessing a specific example from memory
        during language learning.
        
        Args:
            idx: Index of sentence pair
        
        Returns:
            src_indices: Source sentence as integer tensor (includes <SOS> and <EOS>)
            tgt_indices: Target sentence as integer tensor (includes <SOS> and <EOS>)
        """
        src_tokens = self.src_sentences[idx]
        tgt_tokens = self.tgt_sentences[idx]
        
        # Encode source sentence
        # Add <SOS> and <EOS> tokens to mark sentence boundaries
        # These markers are like prosodic cues that signal utterance boundaries
        src_indices = [self.src_vocab.SOS_IDX] + \
                     self.src_vocab.encode(src_tokens) + \
                     [self.src_vocab.EOS_IDX]
        
        # Encode target sentence
        tgt_indices = [self.tgt_vocab.SOS_IDX] + \
                     self.tgt_vocab.encode(tgt_tokens) + \
                     [self.tgt_vocab.EOS_IDX]
        
        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(tgt_indices, dtype=torch.long)
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Collate batch of sentence pairs with padding.
        
        This batching process is like organizing multiple learning examples
        for efficient parallel processing, similar to how classrooms handle
        multiple students simultaneously.
        
        Padding is necessary because sentences have variable lengths, but neural
        networks require fixed-size inputs. We pad shorter sentences with <PAD>
        tokens and create masks to ignore these positions during processing.
        
        Args:
            batch: List of (src_indices, tgt_indices) tuples
        
        Returns:
            src_batch: [batch_size, max_src_len] - padded source sequences
            src_lengths: [batch_size] - actual source lengths (before padding)
            tgt_batch: [batch_size, max_tgt_len] - padded target sequences
            tgt_lengths: [batch_size] - actual target lengths (before padding)
        """
        # Separate source and target sequences
        src_sequences = [item[0] for item in batch]
        tgt_sequences = [item[1] for item in batch]
        
        # Get actual lengths (before padding)
        # These lengths are important for ignoring padding during processing
        src_lengths = torch.tensor([len(seq) for seq in src_sequences], dtype=torch.long)
        tgt_lengths = torch.tensor([len(seq) for seq in tgt_sequences], dtype=torch.long)
        
        # Pad sequences to same length within batch
        # pad_sequence adds <PAD> tokens (index 0) to make all sequences same length
        # This is like filling in blank spaces so all examples have the same format
        src_batch = pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=0  # <PAD> token index
        )  # [batch_size, max_src_len]
        
        tgt_batch = pad_sequence(
            tgt_sequences,
            batch_first=True,
            padding_value=0  # <PAD> token index
        )  # [batch_size, max_tgt_len]
        
        return src_batch, src_lengths, tgt_batch, tgt_lengths


def create_masks(src_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create mask for padded positions.
    
    Masks tell the model which positions are real tokens (1) and which are
    padding (0). This is like knowing which parts of input to pay attention to
    and which to ignore.
    
    Args:
        src_lengths: [batch_size] - actual lengths of sequences
        max_len: Maximum length in the batch
    
    Returns:
        mask: [batch_size, max_len] - 1 for real tokens, 0 for padding
    
    Example:
        >>> lengths = torch.tensor([5, 3, 4])
        >>> mask = create_masks(lengths, max_len=5)
        >>> mask
        tensor([[1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0]])
    """
    batch_size = src_lengths.size(0)
    mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, length in enumerate(src_lengths):
        mask[i, :length] = 1
    
    return mask

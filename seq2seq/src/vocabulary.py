"""
Vocabulary Management for Neural Machine Translation

This module implements vocabulary management for seq2seq translation, handling
the mapping between words and integer indices. This relates to semantic memory
in cognitive science - how the brain represents and organizes word meanings.

Cognitive Science Connection:
- Vocabulary represents semantic memory: the mental lexicon of word meanings
- Word-to-index mapping parallels how the brain encodes concepts as neural patterns
- Special tokens (<PAD>, <SOS>, <EOS>, <UNK>) mirror linguistic markers in human language
"""

from collections import Counter
from typing import List, Optional


class Vocabulary:
    """
    Manages word-to-index and index-to-word mappings for a language.
    
    Special Tokens:
        <PAD> (index 0): Padding token for variable-length sequences
        <SOS> (index 1): Start-of-sequence marker
        <EOS> (index 2): End-of-sequence marker
        <UNK> (index 3): Unknown word token for out-of-vocabulary words
    
    These special tokens parallel linguistic markers in human language:
    - <SOS>/<EOS> mark utterance boundaries (like prosodic cues)
    - <UNK> represents the concept of "unknown word" (like when we hear unfamiliar terms)
    - <PAD> is a computational necessity (no direct cognitive parallel)
    """
    
    # Special token constants
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3
    
    def __init__(self):
        """Initialize vocabulary with special tokens."""
        # Word to index mapping
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
        }
        
        # Index to word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Track next available index
        self.next_idx = 4
    
    def build_vocab(
        self,
        sentences: List[List[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None
    ) -> None:
        """
        Build vocabulary from tokenized sentences.
        
        This process mirrors how humans build their mental lexicon through
        exposure to language. Words that appear frequently become part of
        our active vocabulary, while rare words may not be learned.
        
        Args:
            sentences: List of tokenized sentences (list of word lists)
            min_freq: Minimum frequency for word inclusion (filters rare words)
            max_size: Maximum vocabulary size (None for unlimited)
        
        Example:
            >>> vocab = Vocabulary()
            >>> sentences = [["hello", "world"], ["hello", "there"]]
            >>> vocab.build_vocab(sentences, min_freq=1)
            >>> len(vocab)  # 4 special tokens + 3 unique words = 7
            7
        """
        # Count word frequencies
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)
        
        # Sort by frequency (most common first)
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary
        for word, freq in sorted_words:
            # Skip if below minimum frequency
            if freq < min_freq:
                continue
            
            # Skip if already in vocabulary
            if word in self.word2idx:
                continue
            
            # Stop if reached maximum size
            if max_size is not None and len(self.word2idx) >= max_size:
                break
            
            # Add word to vocabulary
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1
    
    def encode(self, sentence: List[str]) -> List[int]:
        """
        Convert tokenized sentence to integer sequence.
        
        This encoding process parallels how the brain transforms linguistic input
        into internal representations. Words are mapped to indices that can be
        processed by the neural network, similar to how concepts activate specific
        neural patterns in the brain.
        
        Args:
            sentence: List of word tokens
        
        Returns:
            List of integer indices
        
        Example:
            >>> vocab = Vocabulary()
            >>> vocab.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "hello": 4}
            >>> vocab.encode(["hello", "world"])
            [4, 3]  # "world" maps to <UNK> if not in vocabulary
        """
        return [self.word2idx.get(word, self.UNK_IDX) for word in sentence]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert integer sequence back to tokenized sentence.
        
        This decoding process parallels language production, where internal
        representations are transformed back into words. The brain performs
        a similar process when converting thoughts into speech.
        
        Args:
            indices: List of integer indices
        
        Returns:
            List of word tokens
        
        Example:
            >>> vocab = Vocabulary()
            >>> vocab.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "hello"}
            >>> vocab.decode([4, 3, 2])
            ["hello", "<UNK>", "<EOS>"]
        """
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2idx)
    
    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2idx
    
    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return f"Vocabulary(size={len(self)}, next_idx={self.next_idx})"

"""
Encoder Network for Sequence-to-Sequence Translation

This module implements the encoder component of the seq2seq architecture,
which processes source language sentences into continuous vector representations.

Cognitive Science Connection:
The encoder mirrors working memory encoding in human cognition. When we hear or
read a sentence, we don't store the exact words but create an abstract semantic
representation that captures meaning. The encoder performs a similar transformation,
compressing variable-length input into fixed-size context vectors.

Key Parallels:
- Sequential processing: Like humans reading word-by-word
- Hidden state accumulation: Like building up understanding incrementally
- Fixed-size representation: Like working memory's limited capacity
- Semantic compression: Capturing meaning rather than surface form
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


class Encoder(nn.Module):
    """
    Encoder network that transforms source sentences into context representations.
    
    Architecture:
        1. Embedding layer: Maps word indices to dense vectors
        2. Recurrent layer: LSTM or GRU processes sequence
        3. Output: Hidden states for all positions + final state
    
    The encoder's hidden state acts like working memory, maintaining a compressed
    representation of everything seen so far in the sentence.
    
    Args:
        vocab_size: Size of source vocabulary
        embedding_dim: Dimension of word embeddings (128-512)
        hidden_dim: Dimension of LSTM/GRU hidden state (256-512)
        cell_type: 'LSTM' or 'GRU' (default: 'LSTM')
        dropout: Dropout probability (default: 0.0 for small datasets)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        cell_type: str = 'LSTM',
        dropout: float = 0.0
    ):
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        
        # Embedding layer: converts word indices to dense vectors
        # This parallels how the brain represents words as distributed patterns
        # of neural activation rather than discrete symbols
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> token index
        )
        
        # Recurrent layer: processes sequence and maintains hidden state
        # LSTM/GRU cells maintain information over time, similar to how
        # working memory maintains context during sentence comprehension
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,  # Single layer for educational clarity
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
        else:
            raise ValueError(f"cell_type must be 'LSTM' or 'GRU', got {cell_type}")
    
    def forward(
        self,
        src_tokens: torch.Tensor,
        src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode source sentences into context representations.
        
        This forward pass mirrors how humans process language incrementally:
        1. Each word is recognized (embedding lookup)
        2. Meaning accumulates as we read (RNN hidden state updates)
        3. Final understanding captured in final hidden state
        
        Args:
            src_tokens: [batch_size, src_seq_len] - tokenized source sentences
            src_lengths: [batch_size] - actual lengths before padding
        
        Returns:
            encoder_outputs: [batch_size, src_seq_len, hidden_dim] - all hidden states
            final_hidden: [batch_size, hidden_dim] - final hidden state
            final_cell: [batch_size, hidden_dim] - final cell state (LSTM only, None for GRU)
        
        Example:
            >>> encoder = Encoder(vocab_size=1000, embedding_dim=256, hidden_dim=512)
            >>> src_tokens = torch.randint(0, 1000, (32, 20))  # batch_size=32, seq_len=20
            >>> src_lengths = torch.randint(5, 20, (32,))
            >>> outputs, hidden, cell = encoder(src_tokens, src_lengths)
            >>> outputs.shape
            torch.Size([32, 20, 512])
            >>> hidden.shape
            torch.Size([32, 512])
        """
        batch_size = src_tokens.size(0)
        
        # Step 1: Embed source tokens
        # Transform discrete word indices into continuous semantic vectors
        # This is like activating semantic representations in the brain
        embedded = self.embedding(src_tokens)  # [batch_size, src_seq_len, embedding_dim]
        
        # Step 2: Pack padded sequences for efficient processing
        # This tells the RNN to ignore padding tokens, similar to how humans
        # don't process "empty" spaces in text
        packed_embedded = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Step 3: Process through recurrent layer
        # The RNN maintains a hidden state that accumulates information,
        # similar to how working memory maintains context during comprehension
        if self.cell_type == 'LSTM':
            packed_outputs, (final_hidden, final_cell) = self.rnn(packed_embedded)
            # Extract final states from (num_layers, batch_size, hidden_dim) to (batch_size, hidden_dim)
            final_hidden = final_hidden.squeeze(0)  # [batch_size, hidden_dim]
            final_cell = final_cell.squeeze(0)      # [batch_size, hidden_dim]
        else:  # GRU
            packed_outputs, final_hidden = self.rnn(packed_embedded)
            final_hidden = final_hidden.squeeze(0)  # [batch_size, hidden_dim]
            final_cell = None
        
        # Step 4: Unpack outputs
        # Restore the padded sequence format for attention mechanism
        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )  # [batch_size, src_seq_len, hidden_dim]
        
        return encoder_outputs, final_hidden, final_cell
    
    def get_output_dim(self) -> int:
        """Return the dimension of encoder outputs (hidden_dim)."""
        return self.hidden_dim

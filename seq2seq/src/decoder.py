"""
Decoder Network for Sequence-to-Sequence Translation

This module implements the decoder component of the seq2seq architecture,
which generates target language sentences word-by-word, conditioned on the
source sentence representation.

Cognitive Science Connection:
The decoder mirrors incremental language production in humans. We generate speech
or writing word-by-word, with each word influenced by:
1. What we've already said (decoder hidden state)
2. What we intend to communicate (encoder context)
3. What we're currently focusing on (attention mechanism)

Key Parallels:
- Autoregressive generation: Like producing speech one word at a time
- State maintenance: Like tracking discourse context during production
- Attention-guided production: Like looking back at notes while writing
- Probabilistic selection: Like choosing between synonyms based on context
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Decoder(nn.Module):
    """
    Decoder network that generates target sentences word-by-word.
    
    Architecture:
        1. Embedding layer: Maps target word indices to dense vectors
        2. Attention mechanism: Focuses on relevant source words
        3. Recurrent layer: Maintains generation state
        4. Output layer: Predicts next word probabilities
    
    The decoder's hidden state acts like a production buffer in language generation,
    maintaining context about what has been generated and what needs to be said next.
    
    Args:
        vocab_size: Size of target vocabulary
        embedding_dim: Dimension of word embeddings (128-512)
        hidden_dim: Dimension of LSTM/GRU hidden state (256-512)
        attention: Attention mechanism instance
        cell_type: 'LSTM' or 'GRU' (default: 'LSTM')
        dropout: Dropout probability (default: 0.0 for small datasets)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        attention: nn.Module,
        cell_type: str = 'LSTM',
        dropout: float = 0.0
    ):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.cell_type = cell_type
        self.attention = attention
        
        # Embedding layer: converts target word indices to dense vectors
        # Like activating semantic representations for words we want to produce
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> token index
        )
        
        # Recurrent layer: processes embedded word + attention context
        # Input is concatenation of word embedding and attention context vector
        # This mirrors how language production integrates:
        # - The word we just said (embedding)
        # - What we're focusing on from the source (context)
        rnn_input_dim = embedding_dim + hidden_dim
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,  # Single layer for educational clarity
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0
            )
        else:
            raise ValueError(f"cell_type must be 'LSTM' or 'GRU', got {cell_type}")
        
        # Output layer: projects hidden state to vocabulary distribution
        # This is like selecting from our mental lexicon which word to produce next
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Perform one decoding step.
        
        This single-step generation mirrors how humans produce language incrementally:
        1. Activate representation of current word (embedding)
        2. Focus on relevant source information (attention)
        3. Update production state (RNN)
        4. Select next word (output projection)
        
        Args:
            tgt_token: [batch_size] - current target token index
            hidden: [batch_size, hidden_dim] - previous decoder hidden state
            cell: [batch_size, hidden_dim] - previous cell state (LSTM only, None for GRU)
            encoder_outputs: [batch_size, src_seq_len, hidden_dim] - all encoder states
            src_mask: [batch_size, src_seq_len] - mask for padding positions
        
        Returns:
            output: [batch_size, vocab_size] - probability distribution over vocabulary
            hidden: [batch_size, hidden_dim] - updated hidden state
            cell: [batch_size, hidden_dim] - updated cell state (LSTM only, None for GRU)
            attention_weights: [batch_size, src_seq_len] - attention weights for this step
        
        Example:
            >>> from attention import BahdanauAttention
            >>> attention = BahdanauAttention(hidden_dim=512)
            >>> decoder = Decoder(vocab_size=5000, embedding_dim=256, hidden_dim=512, attention=attention)
            >>> tgt_token = torch.randint(0, 5000, (32,))  # batch_size=32
            >>> hidden = torch.randn(32, 512)
            >>> cell = torch.randn(32, 512)
            >>> encoder_outputs = torch.randn(32, 20, 512)
            >>> src_mask = torch.ones(32, 20)
            >>> output, hidden, cell, attn_weights = decoder(tgt_token, hidden, cell, encoder_outputs, src_mask)
            >>> output.shape
            torch.Size([32, 5000])
            >>> attn_weights.sum(dim=1)  # Should sum to 1.0
            tensor([1., 1., 1., ...])
        """
        batch_size = tgt_token.size(0)
        
        # Step 1: Embed current target token
        # Transform discrete word index into continuous semantic vector
        # Like activating the meaning of the word we're about to produce
        embedded = self.embedding(tgt_token)  # [batch_size, embedding_dim]
        
        # Step 2: Compute attention context vector
        # Focus on relevant parts of the source sentence
        # This is like a translator looking back at specific source words
        context_vector, attention_weights = self.attention(
            hidden, encoder_outputs, src_mask
        )  # context: [batch_size, hidden_dim], weights: [batch_size, src_seq_len]
        
        # Step 3: Concatenate embedded token with attention context
        # Combine what we're saying with what we're focusing on
        # This integration mirrors how humans use both internal plans and
        # external information when producing language
        rnn_input = torch.cat([embedded, context_vector], dim=1)  # [batch_size, embedding_dim + hidden_dim]
        rnn_input = rnn_input.unsqueeze(1)  # [batch_size, 1, embedding_dim + hidden_dim]
        
        # Step 4: Update decoder state through RNN
        # Maintain context about what we've generated so far
        # Like updating our mental model of the discourse
        if self.cell_type == 'LSTM':
            # LSTM maintains both hidden state (short-term) and cell state (long-term)
            hidden_input = hidden.unsqueeze(0)  # [1, batch_size, hidden_dim]
            cell_input = cell.unsqueeze(0)      # [1, batch_size, hidden_dim]
            rnn_output, (hidden_new, cell_new) = self.rnn(rnn_input, (hidden_input, cell_input))
            hidden_new = hidden_new.squeeze(0)  # [batch_size, hidden_dim]
            cell_new = cell_new.squeeze(0)      # [batch_size, hidden_dim]
        else:  # GRU
            hidden_input = hidden.unsqueeze(0)  # [1, batch_size, hidden_dim]
            rnn_output, hidden_new = self.rnn(rnn_input, hidden_input)
            hidden_new = hidden_new.squeeze(0)  # [batch_size, hidden_dim]
            cell_new = None
        
        rnn_output = rnn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Step 5: Project to vocabulary to get next word probabilities
        # Select from our mental lexicon which word to produce
        # The model learns which words are appropriate given the current state
        output = self.output_projection(rnn_output)  # [batch_size, vocab_size]
        
        return output, hidden_new, cell_new, attention_weights
    
    def init_hidden(self, encoder_final_hidden: torch.Tensor) -> torch.Tensor:
        """
        Initialize decoder hidden state from encoder's final state.
        
        This transfer of state is like how we maintain the meaning of a source
        sentence in working memory while beginning to produce the translation.
        
        Args:
            encoder_final_hidden: [batch_size, hidden_dim] - encoder's final hidden state
        
        Returns:
            Initial decoder hidden state (same as encoder_final_hidden)
        """
        return encoder_final_hidden
    
    def init_cell(self, encoder_final_cell: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Initialize decoder cell state from encoder's final cell state (LSTM only).
        
        Args:
            encoder_final_cell: [batch_size, hidden_dim] - encoder's final cell state
        
        Returns:
            Initial decoder cell state (same as encoder_final_cell, or None for GRU)
        """
        return encoder_final_cell

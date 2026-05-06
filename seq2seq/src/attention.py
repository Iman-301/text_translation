"""
Attention Mechanism for Sequence-to-Sequence Translation

This module implements the Bahdanau attention mechanism, which allows the decoder
to selectively focus on relevant parts of the source sentence when generating
each target word.

Cognitive Science Connection:
Attention mechanism directly parallels selective attention in cognitive psychology.
When translating, humans don't process the entire source sentence uniformly but
focus on specific words or phrases relevant to the current target word being produced.

Key Parallels:
- Selective focus: Like visual or auditory attention in perception
- Dynamic weighting: Attention shifts as translation progresses
- Context integration: Combining focused information with current state
- Limited capacity: Can't attend to everything simultaneously

This is one of the most cognitively plausible components of neural translation,
as it mirrors eye-tracking studies showing translators look back at specific
source words when producing each target word.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    This attention computes alignment scores between the decoder's current state
    and all encoder hidden states, then uses these scores to create a weighted
    combination (context vector) of encoder outputs.
    
    The attention weights show what the model is "looking at" in the source
    sentence, similar to how eye-tracking reveals where humans focus during
    translation.
    
    Formula:
        score(h_t, h_s) = v^T * tanh(W1 * h_t + W2 * h_s)
        attention_weights = softmax(scores)
        context = sum(attention_weights * encoder_outputs)
    
    Args:
        hidden_dim: Dimension of encoder and decoder hidden states
    """
    
    def __init__(self, hidden_dim: int):
        super(BahdanauAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Linear layers for computing attention scores
        # W1: projects decoder hidden state
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # W2: projects encoder hidden states
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # v: final projection to scalar score
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention context vector and weights.
        
        This process mirrors how humans selectively attend to relevant information:
        1. Compare current state with all available information (score computation)
        2. Determine what's most relevant (softmax normalization)
        3. Focus on relevant information (weighted combination)
        
        Args:
            decoder_hidden: [batch_size, hidden_dim] - current decoder state
            encoder_outputs: [batch_size, src_seq_len, hidden_dim] - all encoder states
            src_mask: [batch_size, src_seq_len] - mask for padding (1 for real tokens, 0 for padding)
        
        Returns:
            context_vector: [batch_size, hidden_dim] - weighted combination of encoder outputs
            attention_weights: [batch_size, src_seq_len] - normalized attention scores
        
        Example:
            >>> attention = BahdanauAttention(hidden_dim=512)
            >>> decoder_hidden = torch.randn(32, 512)  # batch_size=32
            >>> encoder_outputs = torch.randn(32, 20, 512)  # seq_len=20
            >>> src_mask = torch.ones(32, 20)
            >>> context, weights = attention(decoder_hidden, encoder_outputs, src_mask)
            >>> context.shape
            torch.Size([32, 512])
            >>> weights.shape
            torch.Size([32, 20])
            >>> weights.sum(dim=1)  # Should sum to 1.0
            tensor([1., 1., 1., ...])
        """
        batch_size = encoder_outputs.size(0)
        src_seq_len = encoder_outputs.size(1)
        
        # Step 1: Project decoder hidden state
        # Expand decoder_hidden to match encoder_outputs shape for broadcasting
        # [batch_size, hidden_dim] -> [batch_size, 1, hidden_dim] -> [batch_size, src_seq_len, hidden_dim]
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        decoder_proj = decoder_proj.expand(-1, src_seq_len, -1)     # [batch_size, src_seq_len, hidden_dim]
        
        # Step 2: Project encoder hidden states
        # [batch_size, src_seq_len, hidden_dim] -> [batch_size, src_seq_len, hidden_dim]
        encoder_proj = self.W_encoder(encoder_outputs)
        
        # Step 3: Compute alignment scores
        # Combine decoder and encoder projections with tanh activation
        # This non-linear combination allows the model to learn complex alignment patterns
        combined = torch.tanh(decoder_proj + encoder_proj)  # [batch_size, src_seq_len, hidden_dim]
        
        # Project to scalar scores
        scores = self.v(combined).squeeze(2)  # [batch_size, src_seq_len]
        
        # Step 4: Apply mask to ignore padding positions
        # Set scores for padding positions to very negative value so they get ~0 attention
        # This is like humans ignoring blank spaces - they don't contribute to understanding
        scores = scores.masked_fill(src_mask == 0, float('-inf'))
        
        # Step 5: Normalize scores to get attention weights
        # Softmax ensures weights sum to 1.0, creating a probability distribution
        # This is like deciding how much to focus on each source word (limited attention capacity)
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, src_seq_len]
        
        # Step 6: Compute context vector as weighted sum of encoder outputs
        # This combines information from all source positions, weighted by relevance
        # Similar to how humans integrate information from multiple words when translating
        # [batch_size, src_seq_len, 1] * [batch_size, src_seq_len, hidden_dim]
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, src_seq_len]
            encoder_outputs                   # [batch_size, src_seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        return context_vector, attention_weights
    
    def visualize_attention_pattern(
        self,
        attention_weights: torch.Tensor,
        src_tokens: list,
        tgt_token: str
    ) -> str:
        """
        Create a simple text visualization of attention weights.
        
        This helps understand what the model is focusing on, similar to
        eye-tracking visualizations in human translation studies.
        
        Args:
            attention_weights: [src_seq_len] - attention weights for one example
            src_tokens: List of source tokens
            tgt_token: Current target token being generated
        
        Returns:
            String visualization of attention pattern
        
        Example:
            >>> attention = BahdanauAttention(512)
            >>> weights = torch.tensor([0.1, 0.6, 0.2, 0.1])
            >>> src = ["the", "cat", "is", "black"]
            >>> attention.visualize_attention_pattern(weights, src, "chat")
            'Generating "chat" - attending to: the(0.10) cat(0.60) is(0.20) black(0.10)'
        """
        pairs = [(token, weight.item()) for token, weight in zip(src_tokens, attention_weights)]
        attention_str = " ".join([f"{token}({weight:.2f})" for token, weight in pairs])
        return f'Generating "{tgt_token}" - attending to: {attention_str}'

"""
Inference Module for Seq2Seq Translation

This module handles translation of new sentences using the trained model,
implementing both greedy decoding and beam search strategies.

Cognitive Science Connection:
Translation generation mirrors spontaneous language production in humans.
Unlike training (where we have the correct answer), inference requires the
model to generate translations autonomously, similar to how bilinguals
produce translations without explicit guidance.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


class Translator:
    """
    Translator for generating translations using trained seq2seq model.
    
    This class implements inference strategies that mirror different approaches
    to language production:
    - Greedy decoding: Like speaking the first word that comes to mind
    - Beam search: Like considering multiple phrasings before choosing the best
    
    Args:
        encoder: Trained encoder network
        decoder: Trained decoder network
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: 'cpu' or 'cuda'
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_vocab,
        tgt_vocab,
        device: str = 'cpu'
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        
        # Set to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
    
    def translate(
        self,
        sentence: str,
        max_length: int = 50,
        method: str = 'greedy',
        beam_width: int = 5
    ) -> Tuple[str, np.ndarray]:
        """
        Translate a source sentence.
        
        This is the main entry point for translation, handling tokenization,
        encoding, decoding, and detokenization.
        
        Args:
            sentence: Source sentence string
            max_length: Maximum translation length
            method: 'greedy' or 'beam_search'
            beam_width: Beam width for beam search (default: 5)
        
        Returns:
            translation: Target sentence string
            attention_weights: [tgt_len, src_len] - attention matrix for visualization
        
        Example:
            >>> translator = Translator(encoder, decoder, src_vocab, tgt_vocab)
            >>> translation, attention = translator.translate("Hello world")
            >>> print(translation)
            'Bonjour le monde'
        """
        # Tokenize source sentence
        src_tokens = sentence.lower().strip().split()
        
        # Encode to indices
        src_indices = [self.src_vocab.SOS_IDX] + \
                     self.src_vocab.encode(src_tokens) + \
                     [self.src_vocab.EOS_IDX]
        
        # Convert to tensor
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
        src_length = torch.tensor([len(src_indices)], dtype=torch.long).to(self.device)
        
        # Translate
        with torch.no_grad():
            if method == 'greedy':
                tgt_indices, attention_weights = self.greedy_decode(src_tensor, src_length, max_length)
            elif method == 'beam_search':
                tgt_indices, attention_weights = self.beam_search(src_tensor, src_length, max_length, beam_width)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'beam_search'")
        
        # Decode to words
        tgt_tokens = self.tgt_vocab.decode(tgt_indices)
        
        # Remove special tokens and join
        tgt_tokens = [t for t in tgt_tokens if t not in [
            self.tgt_vocab.SOS_TOKEN,
            self.tgt_vocab.EOS_TOKEN,
            self.tgt_vocab.PAD_TOKEN
        ]]
        translation = ' '.join(tgt_tokens)
        
        return translation, attention_weights
    
    def greedy_decode(
        self,
        src_tensor: torch.Tensor,
        src_length: torch.Tensor,
        max_length: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        Greedy decoding: select highest probability token at each step.
        
        This is like spontaneous speech production where we say the first
        word that comes to mind without considering alternatives. It's fast
        but may not produce the globally best translation.
        
        Args:
            src_tensor: [1, src_seq_len] - source sentence
            src_length: [1] - source length
            max_length: Maximum generation length
        
        Returns:
            tokens: List of generated token indices
            attention_weights: [tgt_len, src_len] - attention matrix
        """
        # Encode source
        encoder_outputs, final_hidden, final_cell = self.encoder(src_tensor, src_length)
        
        # Create source mask
        src_mask = torch.ones(1, src_tensor.size(1), dtype=torch.long, device=self.device)
        
        # Initialize decoder
        decoder_hidden = self.decoder.init_hidden(final_hidden)
        decoder_cell = self.decoder.init_cell(final_cell)
        decoder_input = torch.tensor([self.tgt_vocab.SOS_IDX], device=self.device)
        
        # Generate tokens
        generated_tokens = []
        attention_history = []
        
        for _ in range(max_length):
            # Decode one step
            output, decoder_hidden, decoder_cell, attention_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                decoder_cell,
                encoder_outputs,
                src_mask
            )
            
            # Select token with highest probability
            # This is like choosing the most activated word in our mental lexicon
            predicted_token = output.argmax(dim=1).item()
            
            # Store token and attention
            generated_tokens.append(predicted_token)
            attention_history.append(attention_weights.squeeze(0).cpu().numpy())
            
            # Stop if <EOS> generated
            if predicted_token == self.tgt_vocab.EOS_IDX:
                break
            
            # Use predicted token as next input
            decoder_input = torch.tensor([predicted_token], device=self.device)
        
        # Convert attention history to matrix
        attention_matrix = np.array(attention_history)  # [tgt_len, src_len]
        
        return generated_tokens, attention_matrix
    
    def beam_search(
        self,
        src_tensor: torch.Tensor,
        src_length: torch.Tensor,
        max_length: int,
        beam_width: int = 5
    ) -> Tuple[List[int], np.ndarray]:
        """
        Beam search decoding: maintain top-k hypotheses.
        
        This is like considering multiple possible phrasings before choosing
        the best one. It's slower but often produces better translations by
        avoiding local optima.
        
        Args:
            src_tensor: [1, src_seq_len] - source sentence
            src_length: [1] - source length
            max_length: Maximum generation length
            beam_width: Number of hypotheses to maintain
        
        Returns:
            tokens: List of generated token indices (best hypothesis)
            attention_weights: [tgt_len, src_len] - attention matrix
        """
        # Encode source
        encoder_outputs, final_hidden, final_cell = self.encoder(src_tensor, src_length)
        
        # Create source mask
        src_mask = torch.ones(1, src_tensor.size(1), dtype=torch.long, device=self.device)
        
        # Initialize beam
        # Each hypothesis: (score, tokens, hidden, cell, attention_history)
        initial_hidden = self.decoder.init_hidden(final_hidden)
        initial_cell = self.decoder.init_cell(final_cell)
        
        beams = [(
            0.0,  # score
            [self.tgt_vocab.SOS_IDX],  # tokens
            initial_hidden,  # hidden state
            initial_cell,  # cell state
            []  # attention history
        )]
        
        completed_beams = []
        
        for step in range(max_length):
            candidates = []
            
            for score, tokens, hidden, cell, attn_history in beams:
                # Skip if already completed
                if tokens[-1] == self.tgt_vocab.EOS_IDX:
                    completed_beams.append((score, tokens, attn_history))
                    continue
                
                # Decode one step
                decoder_input = torch.tensor([tokens[-1]], device=self.device)
                output, new_hidden, new_cell, attention_weights = self.decoder(
                    decoder_input,
                    hidden,
                    cell,
                    encoder_outputs,
                    src_mask
                )
                
                # Get top-k tokens
                log_probs = torch.log_softmax(output, dim=1)
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                
                # Create new hypotheses
                for i in range(beam_width):
                    new_token = top_indices[0, i].item()
                    new_score = score + top_log_probs[0, i].item()
                    new_tokens = tokens + [new_token]
                    new_attn_history = attn_history + [attention_weights.squeeze(0).cpu().numpy()]
                    
                    candidates.append((
                        new_score,
                        new_tokens,
                        new_hidden,
                        new_cell,
                        new_attn_history
                    ))
            
            # Select top beam_width hypotheses
            # Sort by score (higher is better)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            # Stop if all beams completed
            if len(completed_beams) >= beam_width:
                break
        
        # Add remaining beams to completed
        for score, tokens, _, _, attn_history in beams:
            completed_beams.append((score, tokens, attn_history))
        
        # Select best hypothesis
        if completed_beams:
            completed_beams.sort(key=lambda x: x[0], reverse=True)
            best_score, best_tokens, best_attn_history = completed_beams[0]
        else:
            # Fallback if no beam completed
            best_score, best_tokens, _, _, best_attn_history = beams[0]
        
        # Remove <SOS> token
        if best_tokens[0] == self.tgt_vocab.SOS_IDX:
            best_tokens = best_tokens[1:]
            if best_attn_history:
                best_attn_history = best_attn_history[1:]
        
        # Convert attention history to matrix
        if best_attn_history:
            attention_matrix = np.array(best_attn_history)
        else:
            attention_matrix = np.zeros((len(best_tokens), src_tensor.size(1)))
        
        return best_tokens, attention_matrix

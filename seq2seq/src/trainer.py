"""
Training Module for Seq2Seq Translation

This module orchestrates the training process including forward/backward passes,
optimization, validation, and checkpointing.

Cognitive Science Connection:
The training process mirrors aspects of second language acquisition. Like human
learners, the model gradually improves through exposure to parallel examples,
adjusting its internal representations based on feedback (gradient descent).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import time


class Trainer:
    """
    Trainer for seq2seq translation model.
    
    This class manages the training loop, similar to how a language teacher
    guides students through learning:
    - Presents examples (forward pass)
    - Provides feedback (loss computation)
    - Adjusts understanding (backpropagation)
    - Tracks progress (validation)
    - Saves best performance (checkpointing)
    
    Args:
        encoder: Encoder network
        decoder: Decoder network
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer instance
        criterion: Loss function
        device: 'cpu' or 'cuda'
        grad_clip: Gradient clipping threshold (default: 1.0)
        teacher_forcing_ratio: Probability of using teacher forcing (default: 1.0)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        grad_clip: float = 1.0,
        teacher_forcing_ratio: float = 1.0
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.grad_clip = grad_clip
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        This mirrors one session of language learning, where the learner
        is exposed to multiple examples and adjusts their understanding.
        
        Returns:
            Average training loss for the epoch
        """
        self.encoder.train()
        self.decoder.train()
        
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        # Progress bar for visual feedback
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (src_batch, src_lengths, tgt_batch, tgt_lengths) in enumerate(pbar):
            # Move data to device
            src_batch = src_batch.to(self.device)
            src_lengths = src_lengths.to(self.device)
            tgt_batch = tgt_batch.to(self.device)
            tgt_lengths = tgt_lengths.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss = self._forward_pass(src_batch, src_lengths, tgt_batch, tgt_lengths)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            # This is like preventing overly drastic changes in understanding
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
            
            # Update parameters
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return epoch_loss / num_batches
    
    def _forward_pass(
        self,
        src_batch: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_batch: torch.Tensor,
        tgt_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform forward pass with teacher forcing.
        
        Teacher forcing means using the ground truth previous word as input
        to the decoder, rather than the model's own prediction. This is like
        a language teacher providing the correct word when a student makes
        a mistake, helping them learn faster.
        
        Args:
            src_batch: [batch_size, src_seq_len]
            src_lengths: [batch_size]
            tgt_batch: [batch_size, tgt_seq_len]
            tgt_lengths: [batch_size]
        
        Returns:
            Loss value
        """
        batch_size = src_batch.size(0)
        tgt_seq_len = tgt_batch.size(1)
        
        # Encode source sentences
        encoder_outputs, final_hidden, final_cell = self.encoder(src_batch, src_lengths)
        
        # Create source mask for attention
        src_mask = self._create_mask(src_lengths, src_batch.size(1))
        
        # Initialize decoder state from encoder
        decoder_hidden = self.decoder.init_hidden(final_hidden)
        decoder_cell = self.decoder.init_cell(final_cell)
        
        # Start with <SOS> token
        decoder_input = tgt_batch[:, 0]  # <SOS> tokens
        
        # Accumulate loss
        total_loss = 0
        
        # Decode step by step (teacher forcing)
        for t in range(1, tgt_seq_len):
            # Decode one step
            output, decoder_hidden, decoder_cell, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                decoder_cell,
                encoder_outputs,
                src_mask
            )
            
            # Compute loss for this step
            # Compare predicted distribution with ground truth
            target = tgt_batch[:, t]
            loss = self.criterion(output, target)
            total_loss += loss
            
            # Teacher forcing: use ground truth as next input
            # This is like a teacher providing the correct word
            decoder_input = target
        
        # Average loss over sequence length
        return total_loss / (tgt_seq_len - 1)
    
    def validate(self) -> float:
        """
        Evaluate on validation set.
        
        This is like testing a language learner on unseen examples to
        assess their true understanding, not just memorization.
        
        Returns:
            Average validation loss
        """
        self.encoder.eval()
        self.decoder.eval()
        
        epoch_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src_batch, src_lengths, tgt_batch, tgt_lengths in self.val_loader:
                # Move data to device
                src_batch = src_batch.to(self.device)
                src_lengths = src_lengths.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                tgt_lengths = tgt_lengths.to(self.device)
                
                # Forward pass (no gradients needed)
                loss = self._forward_pass(src_batch, src_lengths, tgt_batch, tgt_lengths)
                epoch_loss += loss.item()
        
        return epoch_loss / num_batches
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str,
        early_stopping_patience: int = 5
    ) -> Dict:
        """
        Full training loop with early stopping and checkpointing.
        
        This orchestrates the entire learning process, similar to a
        language course that spans multiple sessions with progress tracking.
        
        Args:
            num_epochs: Maximum number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Stop if no improvement for this many epochs
        
        Returns:
            Dictionary with training history
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Max epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    checkpoint_path / 'best_model.pt',
                    epoch,
                    val_loss
                )
                print(f"  → New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping check
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"No improvement for {early_stopping_patience} epochs")
                break
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    checkpoint_path / f'checkpoint_epoch_{epoch}.pt',
                    epoch,
                    val_loss
                )
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final training loss: {self.train_losses[-1]:.4f}")
        print("=" * 60)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create mask for padded positions."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=self.device).expand(batch_size, max_len)
        mask = (mask < lengths.unsqueeze(1)).long()
        return mask

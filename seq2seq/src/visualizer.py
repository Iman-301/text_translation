"""
Attention Visualization Module

This module generates heatmap visualizations of attention weights, showing
which source words the model focuses on when generating each target word.

Cognitive Science Connection:
Attention visualizations are analogous to eye-tracking studies in human
translation research. They reveal the model's "focus of attention" during
translation, similar to how eye-tracking shows where translators look
when producing each word.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List


class AttentionVisualizer:
    """
    Visualizer for attention weight heatmaps.
    
    These visualizations help us understand what the model is "looking at"
    during translation, providing insights into:
    - Word alignment patterns (which source words map to which target words)
    - Reordering strategies (how word order changes between languages)
    - Attention distribution (focused vs. diffuse attention)
    
    This is similar to eye-tracking visualizations in psycholinguistic research,
    where we track where humans look when reading or translating.
    """
    
    def __init__(self, figsize: tuple = (10, 8), cmap: str = 'YlOrRd'):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
            cmap: Colormap for heatmap (default: 'YlOrRd' - yellow to red)
        """
        self.figsize = figsize
        self.cmap = cmap
        
        # Set style for better-looking plots
        sns.set_style("whitegrid")
    
    def plot_attention(
        self,
        attention_weights: np.ndarray,
        src_tokens: List[str],
        tgt_tokens: List[str],
        save_path: str,
        title: str = "Attention Weights"
    ) -> None:
        """
        Create and save attention heatmap.
        
        The heatmap shows attention weights as color intensity:
        - Bright colors (red/orange): High attention (model is focusing here)
        - Light colors (yellow/white): Low attention (model is ignoring this)
        
        This visualization reveals alignment patterns, similar to how
        eye-tracking heatmaps show where humans focus their gaze.
        
        Args:
            attention_weights: [tgt_len, src_len] - attention matrix
            src_tokens: List of source tokens
            tgt_tokens: List of generated target tokens
            save_path: Path to save the image file
            title: Plot title
        
        Example:
            >>> visualizer = AttentionVisualizer()
            >>> attention = np.random.rand(5, 4)  # 5 target words, 4 source words
            >>> src = ["the", "cat", "is", "black"]
            >>> tgt = ["le", "chat", "est", "noir", "<EOS>"]
            >>> visualizer.plot_attention(attention, src, tgt, "attention.png")
        """
        # Validate dimensions
        if attention_weights.shape[0] != len(tgt_tokens):
            raise ValueError(
                f"Attention matrix has {attention_weights.shape[0]} rows "
                f"but {len(tgt_tokens)} target tokens"
            )
        if attention_weights.shape[1] != len(src_tokens):
            raise ValueError(
                f"Attention matrix has {attention_weights.shape[1]} columns "
                f"but {len(src_tokens)} source tokens"
            )
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        # annot=True shows the numerical values in each cell
        # fmt='.2f' formats numbers to 2 decimal places
        # cbar_kws adds a label to the colorbar
        sns.heatmap(
            attention_weights,
            xticklabels=src_tokens,
            yticklabels=tgt_tokens,
            cmap=self.cmap,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Attention Weight'},
            vmin=0.0,
            vmax=1.0,
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel('Source Words', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target Words', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attention visualization saved to {save_path}")
    
    def plot_multiple_attentions(
        self,
        attention_list: List[tuple],
        save_dir: str,
        prefix: str = "attention"
    ) -> None:
        """
        Create multiple attention visualizations.
        
        This is useful for comparing attention patterns across different
        translations, similar to comparing eye-tracking patterns across
        different participants or conditions.
        
        Args:
            attention_list: List of (attention_weights, src_tokens, tgt_tokens, title) tuples
            save_dir: Directory to save visualizations
            prefix: Filename prefix (default: "attention")
        
        Example:
            >>> visualizer = AttentionVisualizer()
            >>> attentions = [
            ...     (attn1, src1, tgt1, "Translation 1"),
            ...     (attn2, src2, tgt2, "Translation 2")
            ... ]
            >>> visualizer.plot_multiple_attentions(attentions, "visualizations/")
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, (attention_weights, src_tokens, tgt_tokens, title) in enumerate(attention_list):
            filename = f"{prefix}_{i+1}.png"
            filepath = save_path / filename
            self.plot_attention(attention_weights, src_tokens, tgt_tokens, str(filepath), title)
        
        print(f"\nSaved {len(attention_list)} attention visualizations to {save_dir}")
    
    def analyze_attention_pattern(
        self,
        attention_weights: np.ndarray,
        src_tokens: List[str],
        tgt_tokens: List[str]
    ) -> dict:
        """
        Analyze attention patterns and provide insights.
        
        This analysis helps understand the model's translation strategy,
        similar to how researchers analyze eye-tracking data to understand
        reading or translation strategies.
        
        Args:
            attention_weights: [tgt_len, src_len] - attention matrix
            src_tokens: List of source tokens
            tgt_tokens: List of target tokens
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {}
        
        # 1. Attention entropy (how focused vs. diffuse)
        # Low entropy = focused attention (looking at specific words)
        # High entropy = diffuse attention (looking at many words)
        from scipy.stats import entropy
        entropies = [entropy(row) for row in attention_weights]
        analysis['avg_entropy'] = np.mean(entropies)
        analysis['entropy_interpretation'] = (
            "Focused attention" if analysis['avg_entropy'] < 1.0 else "Diffuse attention"
        )
        
        # 2. Alignment pattern (monotonic vs. non-monotonic)
        # Monotonic = source and target word order is similar
        # Non-monotonic = significant reordering
        peak_positions = [np.argmax(row) for row in attention_weights]
        is_monotonic = all(peak_positions[i] <= peak_positions[i+1] 
                          for i in range(len(peak_positions)-1))
        analysis['is_monotonic'] = is_monotonic
        analysis['alignment_pattern'] = "Monotonic" if is_monotonic else "Non-monotonic (reordering)"
        
        # 3. Most attended source words
        total_attention = attention_weights.sum(axis=0)
        top_indices = np.argsort(total_attention)[::-1][:3]
        analysis['most_attended_words'] = [
            (src_tokens[i], total_attention[i]) for i in top_indices
        ]
        
        # 4. Attention coverage (are all source words attended to?)
        min_attention = total_attention.min()
        max_attention = total_attention.max()
        analysis['attention_coverage'] = {
            'min': float(min_attention),
            'max': float(max_attention),
            'interpretation': (
                "Good coverage" if min_attention > 0.1 else 
                "Some words ignored (low coverage)"
            )
        }
        
        return analysis
    
    def print_analysis(self, analysis: dict) -> None:
        """
        Print attention analysis in readable format.
        
        Args:
            analysis: Analysis dictionary from analyze_attention_pattern
        """
        print("\n" + "=" * 60)
        print("Attention Pattern Analysis")
        print("=" * 60)
        print(f"Attention Focus: {analysis['entropy_interpretation']}")
        print(f"  (Average entropy: {analysis['avg_entropy']:.3f})")
        print(f"\nAlignment Pattern: {analysis['alignment_pattern']}")
        print(f"\nMost Attended Source Words:")
        for word, attention in analysis['most_attended_words']:
            print(f"  - '{word}': {attention:.3f}")
        print(f"\nAttention Coverage: {analysis['attention_coverage']['interpretation']}")
        print(f"  (Min: {analysis['attention_coverage']['min']:.3f}, "
              f"Max: {analysis['attention_coverage']['max']:.3f})")
        print("=" * 60)

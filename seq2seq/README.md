# Seq2Seq Neural Machine Translation - From Scratch

A from-scratch implementation of sequence-to-sequence translation with attention mechanism for educational purposes in cognitive science NLP courses.

## Project Overview

This project demonstrates how neural networks can learn to translate between languages while drawing parallels to human cognitive processes:

- **Encoder** → Working memory encoding
- **Decoder** → Language production  
- **Attention** → Selective attention
- **Embeddings** → Semantic memory
- **Training** → Language acquisition

## What's Implemented

### Core Components ✅
- `src/vocabulary.py` - Vocabulary management with special tokens
- `src/encoder.py` - LSTM/GRU encoder network
- `src/attention.py` - Bahdanau attention mechanism
- `src/decoder.py` - LSTM/GRU decoder with attention
- `src/dataset.py` - Data loading and batching
- `src/utils.py` - Helper functions

### Scripts ✅
- `scripts/download_data.py` - Download and prepare Tatoeba dataset

### ✅ All Core Components Implemented!

All essential components are now complete and ready to use:

1. **Training Module** (`src/trainer.py`) ✅
   - Training loop with teacher forcing
   - Validation loop
   - Early stopping
   - Checkpoint saving/loading

2. **Inference Module** (`src/translator.py`) ✅
   - Greedy decoding
   - Beam search
   - Translation generation

3. **Visualization Module** (`src/visualizer.py`) ✅
   - Attention heatmap generation
   - Attention pattern analysis

4. **Training Script** (`scripts/train.py`) ✅
   - Main training entry point

5. **Translation Script** (`scripts/translate.py`) ✅
   - Command-line translation tool

6. **Visualization Script** (`scripts/visualize_samples.py`) ✅
   - Generate sample attention visualizations

7. **Documentation** ✅
   - `docs/cognitive_connections.md` - Comprehensive cognitive science explanations

### Optional Enhancements

The following are optional and can be added later:

- Unit tests for each component
- Property-based tests
- Jupyter notebook demo

## Quick Start

### 1. Install Dependencies

```bash
cd seq2seq
pip install -r requirements.txt
```

### 2. Download Data

```bash
python scripts/download_data.py --lang_pair en-fr --num_samples 3000
```

This downloads 3000 English-French sentence pairs from Tatoeba.

### 3. Train Model

```bash
python scripts/train.py --embedding_dim 256 --hidden_dim 512 --epochs 50
```

Training will take approximately 1-2 hours on CPU for 3000 sentence pairs.

### 4. Translate Sentences

```bash
python scripts/translate.py --sentence "Hello world" --checkpoint models/checkpoints/best_model.pt
```

Try different decoding methods:
```bash
python scripts/translate.py --sentence "The cat is black" --method beam_search --beam_width 5
```

### 5. Visualize Attention

```bash
python scripts/visualize_samples.py --checkpoint models/checkpoints/best_model.pt
```

This generates attention heatmaps for 10 sample sentences, saved to `visualizations/`.

## Architecture

```
Source Sentence
      ↓
  [Embedding]
      ↓
   [Encoder] ──→ Hidden States
      ↓              ↓
  Final State    [Attention] ←── Decoder State
      ↓              ↓
  [Decoder] ←── Context Vector
      ↓
  [Output Layer]
      ↓
Target Sentence
```

## Cognitive Science Connections

### Encoder: Working Memory Encoding
The encoder processes source sentences sequentially, maintaining a hidden state that accumulates information. This mirrors how working memory encodes incoming linguistic information into an abstract semantic representation.

### Decoder: Language Production
The decoder generates target sentences word-by-word, similar to incremental language production in humans. Each word is influenced by what's been said and what needs to be communicated.

### Attention: Selective Attention
The attention mechanism allows the decoder to focus on relevant source words, paralleling how humans selectively attend to information during translation.

### Embeddings: Semantic Memory
Word embeddings represent semantic relationships in continuous space, similar to how the brain represents concepts through distributed neural patterns.

## Training Details

- **Dataset**: 3000 English-French sentence pairs from Tatoeba
- **Training Time**: ~1-2 hours on CPU
- **Architecture**: Single-layer LSTM/GRU (educational simplicity)
- **Embedding Dim**: 256
- **Hidden Dim**: 512
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)

## Expected Performance

This is an educational project, not production software. Expected results:

- ✅ Simple sentences: "The cat is black" → "Le chat est noir"
- ✅ Common phrases: "Good morning" → "Bonjour"
- ⚠️ Complex sentences: May have errors but show interesting attention patterns
- ⚠️ Rare words: Will use `<UNK>` token

## Project Structure

```
seq2seq/
├── data/
│   ├── raw/              # Downloaded data
│   ├── processed/        # Preprocessed train/val splits
│   └── vocab/            # Saved vocabularies
├── models/
│   └── checkpoints/      # Model checkpoints
├── visualizations/       # Attention heatmaps
├── src/
│   ├── vocabulary.py     # ✅ Vocabulary management
│   ├── encoder.py        # ✅ LSTM/GRU encoder
│   ├── attention.py      # ✅ Bahdanau attention
│   ├── decoder.py        # ✅ LSTM/GRU decoder
│   ├── dataset.py        # ✅ Data loading
│   ├── utils.py          # ✅ Helper functions
│   ├── trainer.py        # ✅ Training loop
│   ├── translator.py     # ✅ Inference (greedy/beam search)
│   └── visualizer.py     # ✅ Attention visualization
├── scripts/
│   ├── download_data.py  # ✅ Data preparation
│   ├── train.py          # ✅ Training script
│   ├── translate.py      # ✅ Translation script
│   └── visualize_samples.py  # ✅ Visualization script
├── tests/                # ⚠️ Optional (not implemented)
├── docs/
│   └── cognitive_connections.md  # ✅ Cognitive science documentation
└── README.md             # ✅ This file
```

## Comparison with Pretrained Model

Your existing code in `../src/` uses pretrained MarianMT models from Hugging Face. This seq2seq implementation differs in:

| Aspect | Pretrained (../src/) | From Scratch (seq2seq/) |
|--------|---------------------|------------------------|
| **Purpose** | Production-ready translation | Educational understanding |
| **Training** | Pre-trained on millions of sentences | Train yourself on 1000-5000 pairs |
| **Quality** | High accuracy | Basic but functional |
| **Understanding** | Black box | See every component |
| **Cognitive Science** | Limited connection | Explicit parallels |
| **Time** | Instant (download model) | ~2 hours training |

## Next Steps

The system is now complete and ready to use! Here's what you can do:

1. **Download Data**: Run `python scripts/download_data.py --lang_pair en-fr --num_samples 3000`
2. **Train Model**: Run `python scripts/train.py` (takes ~1-2 hours on CPU)
3. **Translate**: Run `python scripts/translate.py --sentence "Your text here"`
4. **Visualize**: Run `python scripts/visualize_samples.py`
5. **Explore**: Read `docs/cognitive_connections.md` for cognitive science insights

### Optional Enhancements

If you want to extend the project:
- Add unit tests (`tests/unit/`)
- Add property-based tests (`tests/property/`)
- Create Jupyter notebook demo (`notebooks/demo.ipynb`)
- Experiment with different hyperparameters
- Try different language pairs
- Compare greedy vs. beam search performance

## References

- Bahdanau et al. (2014): "Neural Machine Translation by Jointly Learning to Align and Translate"
- Sutskever et al. (2014): "Sequence to Sequence Learning with Neural Networks"
- Tatoeba Project: https://tatoeba.org/

## License

Educational use only.

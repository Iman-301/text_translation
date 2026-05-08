# Cognitive Science NLP Project: Text Translation

This project implements a text translation system for cognitive science use cases.
It helps translate research snippets, participant instructions, and study materials
between languages.

## Why this fits Cognitive Science

Translation supports:
- Access to multilingual cognitive science papers and reports.
- Cross-cultural experiment materials and participant communication.
- Language-processing research workflows in NLP and psycholinguistics.

This project also includes an optional **from-scratch Seq2Seq + Attention** implementation (educational) that connects the model components to:
- **Working memory** (encoder state)
- **Selective attention** (attention weights)
- **Incremental language production** (decoder)

## Project Type

- Topic: NLP
- Title: Text Translation
- Approach: From-scratch Seq2Seq+Attention to demonstrate core cognitive/NLP concepts

## Features

- 100% **from-scratch** Seq2Seq + Attention translation system (educational).
- Simple command-line interface.
- Sleek web interface (presentation-friendly, no Gradio).
- Greedy decoding and beam search decoding.

## Tech Stack

- Python 3.10+
- `torch` + `numpy`
- `fastapi` + `uvicorn` for UI/server
- Custom Seq2Seq + Bahdanau attention (`seq2seq/`)

## Quick Demo (Recommended for Presentation)

Start the web UI (always run from the repo root):

```bash
python3 run_web.py
```

Then open `http://127.0.0.1:8000`.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (CLI)

Translate a single sentence:

```bash
python3 run_cli.py --text "how are you" --to fr
python3 run_cli.py --text "how are you" --to es
python3 run_cli.py --text "how are you" --to de
```

## Run (Web UI)

```bash
python3 run_web.py
```

Then open the local URL shown in your terminal.

## Training (required)

This project trains **one multilingual model** for 3 easy target languages (**French/Spanish/German**)
using target-language tags in the input (`<2fr>`, `<2es>`, `<2de>`).

### Download data from Tatoeba (needs internet)

```bash
cd seq2seq
python3 scripts/download_data_multilingual.py --num_samples_per_lang 3000
python3 scripts/train.py --data_dir data/processed_multilingual --epochs 20 --min_freq 2
cd ..
```

After training, the web UI expects:
- `seq2seq/models/checkpoints/best_model.pt`
- `seq2seq/data/vocab/src_vocab.pkl`
- `seq2seq/data/vocab/tgt_vocab.pkl`

## Common Issues

### 1) Running from the wrong folder
Always run from the repo root using:
- `python3 run_web.py`
- `python3 run_cli.py ...`

## Project Structure

```
text_translation/
├─ src/
│  ├─ web_app.py          # FastAPI server (scratch-only)
│  ├─ web_ui/             # Modern single-page UI
│  ├─ scratch_cli.py       # CLI (scratch-only)
│  └─ seq2seq_backend.py   # Loads scratch model + runs inference
├─ seq2seq/                # from-scratch educational implementation
├─ requirements.txt
├─ run_cli.py
├─ run_web.py
└─ README.md
```

## Do you need fine-tuning?

For your course project baseline, no.
Pretrained translation models are enough to demonstrate NLP concepts and cognitive-science relevance.

Fine-tuning is optional if you want better performance on a specific domain (for example, cognitive neuroscience terminology). For that, you would need a parallel dataset (source/target sentence pairs).

## Suggested report sections

1. Problem statement and cognitive-science relevance.
2. From-scratch model (Seq2Seq + Attention) and cognitive parallels.
4. Demo results on cognitive-science sample text + brief error analysis.
5. Limitations and future work (more data, better tokenization, evaluation with BLEU, handling pragmatics).

## 20-minute physical demo script (suggested)

1. **Problem + cognitive science motivation (2 min)**: multilingual research + experiment materials.
2. **System overview (3 min)**: scratch Seq2Seq + Attention and why it relates to cognition.
3. **Live demo: scratch Seq2Seq (12 min)**:
   - Translate 2–3 simple sentences
   - Switch greedy vs beam search
   - Mention attention as a model of selective attention (optionally show attention visualizations from `seq2seq/visualizations/`)
5. **Limitations + future work (3 min)**: domain vocabulary, long sentences, pragmatics, evaluation metrics.

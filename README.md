# Cognitive Science NLP Project: Text Translation

This project implements a text translation system for cognitive science use cases.
It helps translate research snippets, participant instructions, and study materials
between languages.

## Why this fits Cognitive Science

Translation supports:
- Access to multilingual cognitive science papers and reports.
- Cross-cultural experiment materials and participant communication.
- Language-processing research workflows in NLP and psycholinguistics.

## Project Type

- Topic: NLP
- Title: Text Translation
- Approach: Pretrained neural machine translation models (no fine-tuning required for baseline)

## Features

- Translate text between multiple language pairs.
- Simple command-line interface.
- Optional Gradio web interface.
- Batch translation from a text file.

## Tech Stack

- Python 3.10+
- `transformers` + `torch`
- Hugging Face MarianMT models (`Helsinki-NLP/opus-mt-*`)
- Optional `gradio` for UI

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run (CLI)

```powershell
python src/main.py --text "Attention and memory are tightly connected cognitive processes." --src en --tgt ar
```

Batch mode:

```powershell
python src/main.py --input_file data/sample_texts_en.txt --src en --tgt fr
```

## Run (Web UI)

```powershell
python src/app.py
```

Then open the local URL shown in your terminal.

## Supported Language Codes (common)

- `en` English
- `ar` Arabic
- `fr` French
- `de` German
- `es` Spanish
- `it` Italian
- `ru` Russian
- `tr` Turkish
- `zh` Chinese
- `ja` Japanese

Note: Not every source-target pair is available for every model naming pattern. If a direct model is unavailable, the app explains the issue.

## Project Structure

```
cognitive/
├─ data/
│  └─ sample_texts_en.txt
├─ src/
│  ├─ translator.py
│  ├─ main.py
│  └─ app.py
├─ requirements.txt
└─ README.md
```

## Do you need fine-tuning?

For your course project baseline, no.
Pretrained translation models are enough to demonstrate NLP concepts and cognitive-science relevance.

Fine-tuning is optional if you want better performance on a specific domain (for example, cognitive neuroscience terminology). For that, you would need a parallel dataset (source/target sentence pairs).

## Suggested report sections

1. Problem statement and cognitive-science relevance.
2. Model choice (pretrained MarianMT).
3. Method (tokenization, sequence-to-sequence generation).
4. Demo results on sample research text.
5. Limitations and future work (domain adaptation/fine-tuning, evaluation with BLEU/COMET).

"""
Create a tiny offline parallel corpus for quick demos.

Why this exists:
- `scripts/download_data.py` pulls Tatoeba from the internet.
- In labs/presentations, internet may be unavailable or blocked.
- This script generates a small EN→FR toy dataset so the from-scratch
  seq2seq pipeline can be trained end-to-end in minutes.

It writes the same files expected by `scripts/train.py`:
  data/processed/train.src, train.tgt, val.src, val.tgt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


TOY_PAIRS_EN_FR: list[tuple[str, str]] = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("you are welcome", "de rien"),
    ("yes", "oui"),
    ("no", "non"),
    ("please", "s'il vous plaît"),
    ("i am hungry", "j'ai faim"),
    ("i am thirsty", "j'ai soif"),
    ("i am tired", "je suis fatigué"),
    ("how are you", "comment ça va"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment tu t'appelles"),
    ("my name is sara", "je m'appelle sara"),
    ("where is the bathroom", "où sont les toilettes"),
    ("i like coffee", "j'aime le café"),
    ("i like tea", "j'aime le thé"),
    ("i love music", "j'aime la musique"),
    ("this is a book", "c'est un livre"),
    ("this is a pen", "c'est un stylo"),
    ("the cat is black", "le chat est noir"),
    ("the dog is small", "le chien est petit"),
    ("the house is big", "la maison est grande"),
    ("open the door", "ouvre la porte"),
    ("close the window", "ferme la fenêtre"),
    ("i need help", "j'ai besoin d'aide"),
    ("call the police", "appelez la police"),
    ("i do not understand", "je ne comprends pas"),
    ("can you repeat", "peux-tu répéter"),
    ("slowly please", "lentement s'il vous plaît"),
    ("i am a student", "je suis étudiant"),
    ("i study cognitive science", "j'étudie les sciences cognitives"),
    ("memory and attention", "mémoire et attention"),
    ("language processing", "traitement du langage"),
    ("the brain learns patterns", "le cerveau apprend des modèles"),
    ("we use attention", "nous utilisons l'attention"),
    ("translate this sentence", "traduis cette phrase"),
    ("the model predicts words", "le modèle prédit des mots"),
    ("practice makes perfect", "c'est en forgeant qu'on devient forgeron"),
]


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tiny offline dataset for seq2seq training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to write processed files (default: data/processed)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeat the toy pairs to increase word frequency (default: 5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs = list(TOY_PAIRS_EN_FR) * max(1, args.repeats)

    random.seed(args.seed)
    random.shuffle(pairs)

    val_size = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    train_src = [s for s, _ in train_pairs]
    train_tgt = [t for _, t in train_pairs]
    val_src = [s for s, _ in val_pairs]
    val_tgt = [t for _, t in val_pairs]

    _write_lines(output_dir / "train.src", train_src)
    _write_lines(output_dir / "train.tgt", train_tgt)
    _write_lines(output_dir / "val.src", val_src)
    _write_lines(output_dir / "val.tgt", val_tgt)

    print("=" * 60)
    print("Created toy parallel corpus (EN→FR)")
    print("=" * 60)
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print("\nNext:")
    print("  python scripts/train.py --epochs 20 --embedding_dim 64 --hidden_dim 128 --min_freq 1")
    print("  python scripts/translate.py --sentence \"good night\"")


if __name__ == "__main__":
    main()

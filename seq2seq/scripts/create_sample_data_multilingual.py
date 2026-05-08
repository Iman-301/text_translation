"""
Create a tiny offline multilingual parallel corpus for quick demos.

Writes the same files expected by `scripts/train.py`, but for a multilingual model
that uses target-language tags in the SOURCE:
  "<2fr> how are you" -> "comment ça va"
  "<2es> how are you" -> "cómo estás"
  "<2de> how are you" -> "wie geht es dir"

Output:
  data/processed_multilingual/train.src, train.tgt, val.src, val.tgt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


TOY_EN_FR: list[tuple[str, str]] = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("you are welcome", "de rien"),
    ("how are you", "comment ça va"),
    ("i am fine", "je vais bien"),
    ("what is your name", "comment tu t'appelles"),
    ("i study cognitive science", "j'étudie les sciences cognitives"),
    ("memory and attention", "mémoire et attention"),
]

TOY_EN_ES: list[tuple[str, str]] = [
    ("hello", "hola"),
    ("good morning", "buenos días"),
    ("good night", "buenas noches"),
    ("thank you", "gracias"),
    ("you are welcome", "de nada"),
    ("how are you", "cómo estás"),
    ("i am fine", "estoy bien"),
    ("what is your name", "cómo te llamas"),
    ("i study cognitive science", "estudio ciencias cognitivas"),
    ("memory and attention", "memoria y atención"),
]

TOY_EN_DE: list[tuple[str, str]] = [
    ("hello", "hallo"),
    ("good morning", "guten morgen"),
    ("good night", "gute nacht"),
    ("thank you", "danke"),
    ("you are welcome", "bitte"),
    ("how are you", "wie geht es dir"),
    ("i am fine", "mir geht es gut"),
    ("what is your name", "wie heißt du"),
    ("i study cognitive science", "ich studiere kognitionswissenschaft"),
    ("memory and attention", "gedächtnis und aufmerksamkeit"),
]

TAGGED = {
    "fr": ("<2fr>", TOY_EN_FR),
    "es": ("<2es>", TOY_EN_ES),
    "de": ("<2de>", TOY_EN_DE),
}


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Create a tiny multilingual toy dataset for seq2seq training")
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_multilingual",
        help="Directory to write processed files (default: data/processed_multilingual)",
    )
    p.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction (default: 0.2)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--repeats", type=int, default=30, help="Repeat the toy pairs (default: 30)")
    args = p.parse_args()

    pairs: list[tuple[str, str]] = []
    for _lang, (tag, lang_pairs) in TAGGED.items():
        for src, tgt in lang_pairs:
            pairs.append((f"{tag} {src}", tgt))

    pairs = pairs * max(1, args.repeats)
    random.seed(args.seed)
    random.shuffle(pairs)

    val_size = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    out_dir = Path(args.output_dir)
    _write_lines(out_dir / "train.src", [s for s, _ in train_pairs])
    _write_lines(out_dir / "train.tgt", [t for _, t in train_pairs])
    _write_lines(out_dir / "val.src", [s for s, _ in val_pairs])
    _write_lines(out_dir / "val.tgt", [t for _, t in val_pairs])

    print("=" * 60)
    print("Created toy multilingual corpus (EN→{FR,ES,DE} with <2xx> tags)")
    print("=" * 60)
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print("\nNext:")
    print("  python scripts/train.py --data_dir data/processed_multilingual --epochs 10 --embedding_dim 64 --hidden_dim 128 --min_freq 1")


if __name__ == "__main__":
    main()


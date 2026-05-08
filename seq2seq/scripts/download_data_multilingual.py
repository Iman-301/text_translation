"""
Multilingual Data Download and Preparation Script (EN→{FR,ES,DE})

Goal:
- Create ONE training dataset that contains multiple target languages.
- We follow the standard multilingual NMT trick: prepend a target-language tag
  token to the source sentence:
    "<2fr> how are you"  -> "comment ça va"
    "<2es> how are you"  -> "¿cómo estás?"
    "<2de> how are you"  -> "wie geht es dir?"

Outputs (same format expected by scripts/train.py):
  data/processed_multilingual/train.src
  data/processed_multilingual/train.tgt
  data/processed_multilingual/val.src
  data/processed_multilingual/val.tgt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from download_data import download_tatoeba, extract_and_process


LANGS = ["fr", "es", "de"]
TAG_BY_LANG = {"fr": "<2fr>", "es": "<2es>", "de": "<2de>"}


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare multilingual EN→{FR,ES,DE} dataset with target tags")
    p.add_argument("--num_samples_per_lang", type=int, default=3000, help="Samples per language (default: 3000)")
    p.add_argument("--val_split", type=float, default=0.1, help="Validation split per language (default: 0.1)")
    p.add_argument("--min_length", type=int, default=1, help="Min words (default: 1)")
    p.add_argument("--max_length", type=int, default=30, help="Max words (default: 30)")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_multilingual",
        help="Output dir (default: data/processed_multilingual)",
    )
    args = p.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_dir = project_dir / "data" / "raw"

    output_dir = project_dir / args.output_dir
    tmp_base = project_dir / "data" / "_tmp_processed_multilingual"

    all_train_src: list[str] = []
    all_train_tgt: list[str] = []
    all_val_src: list[str] = []
    all_val_tgt: list[str] = []

    for lang in LANGS:
        lang_pair = f"en-{lang}"
        print("=" * 60)
        print(f"Preparing {lang_pair}")
        print("=" * 60)

        zip_path = download_tatoeba(lang_pair, raw_dir)
        tmp_out = tmp_base / lang_pair
        extract_and_process(
            zip_path=zip_path,
            output_dir=tmp_out,
            num_samples=args.num_samples_per_lang,
            min_length=args.min_length,
            max_length=args.max_length,
            val_split=args.val_split,
        )

        tag = TAG_BY_LANG[lang]
        train_src = [f"{tag} {s}".strip() for s in _read_lines(tmp_out / "train.src") if s.strip()]
        train_tgt = [t for t in _read_lines(tmp_out / "train.tgt") if t.strip()]
        val_src = [f"{tag} {s}".strip() for s in _read_lines(tmp_out / "val.src") if s.strip()]
        val_tgt = [t for t in _read_lines(tmp_out / "val.tgt") if t.strip()]

        # Safety: keep aligned lengths
        train_n = min(len(train_src), len(train_tgt))
        val_n = min(len(val_src), len(val_tgt))
        all_train_src.extend(train_src[:train_n])
        all_train_tgt.extend(train_tgt[:train_n])
        all_val_src.extend(val_src[:val_n])
        all_val_tgt.extend(val_tgt[:val_n])

    # Shuffle the combined sets so batches mix languages.
    random.seed(args.seed)
    train_pairs = list(zip(all_train_src, all_train_tgt))
    val_pairs = list(zip(all_val_src, all_val_tgt))
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    train_src_out = [s for s, _ in train_pairs]
    train_tgt_out = [t for _, t in train_pairs]
    val_src_out = [s for s, _ in val_pairs]
    val_tgt_out = [t for _, t in val_pairs]

    _write_lines(output_dir / "train.src", train_src_out)
    _write_lines(output_dir / "train.tgt", train_tgt_out)
    _write_lines(output_dir / "val.src", val_src_out)
    _write_lines(output_dir / "val.tgt", val_tgt_out)

    print("\n" + "=" * 60)
    print("Multilingual data preparation complete!")
    print("=" * 60)
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Train pairs: {len(train_src_out)}")
    print(f"Val pairs:   {len(val_src_out)}")
    print("\nNext:")
    print("  python scripts/train.py --data_dir data/processed_multilingual --min_freq 1 --epochs 20")


if __name__ == "__main__":
    main()


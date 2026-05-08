"""
Prepare a many-to-many parallel corpus with target-language tags (bidirectional).

We create ONE dataset mixing directions by prepending a target tag to the source:

Forward:
  "<2fr> hello" -> "bonjour"
  "<2am> hello" -> "ሰላም"

Reverse (to English):
  "<2en> bonjour" -> "hello"
  "<2en> ሰላም"    -> "hello"

Output (format expected by scripts/train.py):
  data/processed_many2many/train.src
  data/processed_many2many/train.tgt
  data/processed_many2many/val.src
  data/processed_many2many/val.tgt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from download_data import download_tatoeba, extract_and_process


LANGS_DEFAULT = ["fr", "es", "de", "am"]
TAG_BY_LANG: dict[str, str] = {
    "en": "<2en>",
    "fr": "<2fr>",
    "es": "<2es>",
    "de": "<2de>",
    "am": "<2am>",
}


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shuffle_pairs(pairs: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    random.seed(seed)
    random.shuffle(pairs)
    return pairs


def _split_pairs(pairs: list[tuple[str, str]], val_split: float) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    val_size = max(1, int(len(pairs) * val_split))
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]
    return train_pairs, val_pairs


def _load_hf_amharic_pairs(limit: int, seed: int) -> list[tuple[str, str]]:
    """
    Load (en, am) sentence pairs from HuggingFace datasets (OPUS-like corpora).

    We try a small set of known dataset identifiers/configs. If none work,
    we raise a clear error so the user can choose an alternative corpus.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: datasets. Install it with: pip install datasets"
        ) from exc

    candidates: list[tuple[str, str | None]] = [
        ("opus100", "en-am"),
        ("opus_books", "en-am"),
        ("opusparacrawl", "en-am"),
    ]

    last_err: Exception | None = None
    for name, config in candidates:
        try:
            ds = load_dataset(name, config) if config else load_dataset(name)
            # Prefer train split; some corpora might have validation/test only.
            split = "train" if "train" in ds else (list(ds.keys())[0] if len(ds) else "train")
            rows = ds[split]
            # Normalize shapes: opus datasets usually have {"translation": {"en": "...", "am": "..."}}
            pairs: list[tuple[str, str]] = []
            for row in rows:
                tr = row.get("translation") if isinstance(row, dict) else None
                if not isinstance(tr, dict):
                    continue
                en = (tr.get("en") or "").strip()
                am = (tr.get("am") or "").strip()
                if not en or not am:
                    continue
                pairs.append((en, am))
                if len(pairs) >= limit:
                    break

            if not pairs:
                raise RuntimeError(f"{name}:{config} contained no usable en/am pairs")

            random.seed(seed)
            random.shuffle(pairs)
            return pairs
        except Exception as exc:  # try next
            last_err = exc
            continue

    raise RuntimeError(
        "Could not load Amharic-English pairs from HuggingFace datasets. "
        "Tried: opus100/en-am, opus_books/en-am, opusparacrawl/en-am. "
        f"Last error: {last_err}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare many-to-many corpus with tags (bidirectional, incl. Amharic).")
    p.add_argument(
        "--langs",
        nargs="+",
        default=LANGS_DEFAULT,
        help="Target languages to include (default: fr es de am). English is added automatically for reverse examples.",
    )
    p.add_argument("--num_samples_per_lang", type=int, default=3000, help="Samples per non-English language")
    p.add_argument("--val_split", type=float, default=0.1, help="Validation split fraction (default: 0.1)")
    p.add_argument("--min_length", type=int, default=1, help="Min words (default: 1)")
    p.add_argument("--max_length", type=int, default=30, help="Max words (default: 30)")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42)")
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_many2many",
        help="Output dir (default: data/processed_many2many)",
    )
    args = p.parse_args()

    langs = [l.strip().lower() for l in args.langs if l.strip()]
    langs = [l for l in langs if l != "en"]
    for l in langs:
        if l not in TAG_BY_LANG:
            raise SystemExit(f"Unsupported language '{l}'. Supported: {', '.join(sorted(TAG_BY_LANG))}")

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_dir = project_dir / "data" / "raw"
    tmp_base = project_dir / "data" / "_tmp_processed_many2many"
    output_dir = project_dir / args.output_dir

    all_pairs: list[tuple[str, str]] = []

    for lang in langs:
        print("=" * 60)
        print(f"Preparing en-{lang} (+ reverse {lang}-en)")
        print("=" * 60)

        if lang == "am":
            # Load from HF datasets and create our own split.
            pairs = _load_hf_amharic_pairs(limit=args.num_samples_per_lang, seed=args.seed)
            forward = [(f"{TAG_BY_LANG['am']} {en}".strip(), am) for en, am in pairs]
            reverse = [(f"{TAG_BY_LANG['en']} {am}".strip(), en) for en, am in pairs]
            all_pairs.extend(forward)
            all_pairs.extend(reverse)
            continue

        # ManyThings/Tatoeba zip-based pipeline (already implemented in download_data.py)
        lang_pair = f"en-{lang}"
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
        train_src = [s for s in _read_lines(tmp_out / "train.src") if s.strip()]
        train_tgt = [t for t in _read_lines(tmp_out / "train.tgt") if t.strip()]
        val_src = [s for s in _read_lines(tmp_out / "val.src") if s.strip()]
        val_tgt = [t for t in _read_lines(tmp_out / "val.tgt") if t.strip()]

        train_n = min(len(train_src), len(train_tgt))
        val_n = min(len(val_src), len(val_tgt))

        # Forward: <2xx> EN -> XX
        all_pairs.extend([(f"{tag} {train_src[i]}".strip(), train_tgt[i]) for i in range(train_n)])
        all_pairs.extend([(f"{tag} {val_src[i]}".strip(), val_tgt[i]) for i in range(val_n)])

        # Reverse: <2en> XX -> EN
        all_pairs.extend([(f"{TAG_BY_LANG['en']} {train_tgt[i]}".strip(), train_src[i]) for i in range(train_n)])
        all_pairs.extend([(f"{TAG_BY_LANG['en']} {val_tgt[i]}".strip(), val_src[i]) for i in range(val_n)])

    # Shuffle and then re-split globally so batches mix directions and languages.
    all_pairs = _shuffle_pairs(all_pairs, seed=args.seed)
    train_pairs, val_pairs = _split_pairs(all_pairs, val_split=args.val_split)

    _write_lines(output_dir / "train.src", (s for s, _ in train_pairs))
    _write_lines(output_dir / "train.tgt", (t for _, t in train_pairs))
    _write_lines(output_dir / "val.src", (s for s, _ in val_pairs))
    _write_lines(output_dir / "val.tgt", (t for _, t in val_pairs))

    print("\n" + "=" * 60)
    print("Many-to-many data preparation complete!")
    print("=" * 60)
    print(f"Output dir: {output_dir.resolve()}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs:   {len(val_pairs)}")
    print("\nNext:")
    print("  python scripts/train.py --data_dir data/processed_many2many --min_freq 1 --epochs 30")


if __name__ == "__main__":
    main()


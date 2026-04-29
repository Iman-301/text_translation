from __future__ import annotations

import argparse
from pathlib import Path

from translator import TranslationConfig, TranslationError, translate_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Translate text using pretrained MarianMT models."
    )
    parser.add_argument("--src", required=True, help="Source language code, e.g. en")
    parser.add_argument("--tgt", required=True, help="Target language code, e.g. ar")
    parser.add_argument("--text", help="Single text string to translate")
    parser.add_argument("--input_file", help="Path to a text file (one sentence per line)")
    parser.add_argument("--max_length", type=int, default=256, help="Generation max length")
    return parser


def translate_lines(input_path: Path, config: TranslationConfig) -> list[str]:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    for line in lines:
        if not line.strip():
            output.append("")
            continue
        output.append(translate_text(line, config))
    return output


def main() -> None:
    args = build_parser().parse_args()

    config = TranslationConfig(src_lang=args.src, tgt_lang=args.tgt, max_length=args.max_length)

    if not args.text and not args.input_file:
        raise SystemExit("Provide either --text or --input_file")

    try:
        if args.text:
            result = translate_text(args.text, config)
            print(result)
            return

        input_path = Path(args.input_file)
        if not input_path.exists():
            raise SystemExit(f"File not found: {input_path}")

        translations = translate_lines(input_path, config)
        out_path = input_path.with_name(f"{input_path.stem}_{args.src}_to_{args.tgt}.txt")
        out_path.write_text("\n".join(translations), encoding="utf-8")
        print(f"Saved translations to: {out_path}")
    except TranslationError as err:
        raise SystemExit(str(err)) from err


if __name__ == "__main__":
    main()

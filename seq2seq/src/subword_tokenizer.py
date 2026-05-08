from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SentencePieceConfig:
    vocab_size: int = 16000
    model_type: str = "bpe"
    character_coverage: float = 1.0


class SubwordTokenizer:
    """
    Tiny wrapper around SentencePiece used by the scratch seq2seq pipeline.

    We keep it minimal: encode -> pieces, decode -> text.
    """

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency: sentencepiece. Install with: pip install sentencepiece") from exc

        self._sp = spm.SentencePieceProcessor()
        ok = self._sp.Load(str(model_path))
        if not ok:
            raise RuntimeError(f"Could not load sentencepiece model: {model_path}")

    def encode(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        return list(self._sp.EncodeAsPieces(text))

    def decode(self, pieces: Iterable[str]) -> str:
        return self._sp.DecodePieces(list(pieces)).strip()


def train_sentencepiece(
    *,
    input_paths: list[str | Path],
    model_prefix: str | Path,
    cfg: SentencePieceConfig,
) -> tuple[Path, Path]:
    """
    Train and write SentencePiece model files: <prefix>.model and <prefix>.vocab.
    """
    try:
        import sentencepiece as spm  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency: sentencepiece. Install with: pip install sentencepiece") from exc

    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    input_arg = ",".join(str(Path(p)) for p in input_paths)
    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=str(model_prefix),
        vocab_size=int(cfg.vocab_size),
        model_type=str(cfg.model_type),
        character_coverage=float(cfg.character_coverage),
        bos_id=-1,  # we use our own <SOS>/<EOS> tokens
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )

    return model_prefix.with_suffix(".model"), model_prefix.with_suffix(".vocab")


from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import MarianMTModel, MarianTokenizer


@dataclass(frozen=True)
class TranslationConfig:
    src_lang: str
    tgt_lang: str
    max_length: int = 256

    @property
    def model_name(self) -> str:
        return f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"


class TranslationError(Exception):
    pass


@lru_cache(maxsize=16)
def _load_model_and_tokenizer(model_name: str):
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as exc:
        raise TranslationError(
            f"Could not load model '{model_name}'. "
            "This language pair may be unsupported or internet may be unavailable."
        ) from exc


def translate_text(text: str, config: TranslationConfig) -> str:
    if not text.strip():
        raise TranslationError("Input text is empty.")

    tokenizer, model = _load_model_and_tokenizer(config.model_name)

    encoded = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    generated = model.generate(**encoded, max_length=config.max_length)
    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)

    return translated[0]

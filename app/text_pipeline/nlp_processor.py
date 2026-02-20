"""
Text Pipeline — NLP Processor (Stanza)

Handles:
  • Tokenization
  • Lemmatization
  • Named Entity Recognition (NER)
  • Text cleaning / normalisation

Stanza is used instead of spaCy because spaCy's C-extension dependency
(blis) has no binary wheel for Python 3.14 on Windows.
Stanza is pure-Python/PyTorch and works on Python 3.14 immediately.

Model download (one-time, ~400 MB for English):
    import stanza; stanza.download("en")
This is called lazily on first pipeline use and cached afterwards.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

import stanza
from loguru import logger

from app.text_pipeline.schemas import NamedEntity


# ── Stanza pipeline loader (singleton per language) ───────────────────────

@lru_cache(maxsize=8)
def _get_pipeline(lang: str) -> stanza.Pipeline:
    """
    Load (and cache) a Stanza pipeline for *lang*.
    Downloads the model automatically on first use.
    Processors used: tokenize, mwt (where applicable), pos, lemma, ner.
    """
    logger.info(f"[NLP] Loading Stanza pipeline for language='{lang}' …")
    try:
        # Try loading without downloading first
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,ner",
            use_gpu=False,          # CPU-only build
            verbose=False,
            download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        )
    except Exception:
        # Model not present — download it
        logger.info(f"[NLP] Stanza model for '{lang}' not found — downloading…")
        stanza.download(lang, verbose=False)
        nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,ner",
            use_gpu=False,
            verbose=False,
        )
    logger.info(f"[NLP] Stanza pipeline ready for language='{lang}'")
    return nlp


# ── Supported languages (Stanza model names) ─────────────────────────────

# Languages we can run through Stanza NER + lemmatisation.
# For unsupported languages we fall back to English for NLP processing
# while preserving the original text.
STANZA_SUPPORTED: set[str] = {"en", "hi", "zh", "fr", "de", "es", "ar", "ja"}


def _resolve_nlp_lang(detected_lang: str) -> str:
    """Map detected language code to a Stanza model id. Falls back to 'en'."""
    return detected_lang if detected_lang in STANZA_SUPPORTED else "en"


# ── Text cleaning ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise whitespace, strip control characters,
    normalise unicode (NFC).  Does NOT lowercase — lemmatisation
    is more accurate on cased text.
    """
    text = unicodedata.normalize("NFC", text)
    # Remove control chars except normal whitespace
    text = re.sub(r"[^\S\n\r\t ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Main NLP processor ────────────────────────────────────────────────────

def process_text(
    text: str,
    lang: str = "en",
) -> dict:
    """
    Run the full NLP stack on *text*.

    Parameters
    ----------
    text  : raw turn text (speaker label already stripped)
    lang  : BCP-47 language code from the language detector

    Returns
    -------
    {
        "cleaned_text"    : str,
        "lemmatized_text" : str,
        "tokens"          : list[str],
        "entities"        : list[NamedEntity],
    }
    """
    cleaned = clean_text(text)
    nlp_lang = _resolve_nlp_lang(lang)

    try:
        nlp = _get_pipeline(nlp_lang)
        doc = nlp(cleaned)
    except Exception as exc:
        logger.error(f"[NLP] Stanza processing failed: {exc}")
        return {
            "cleaned_text": cleaned,
            "lemmatized_text": cleaned,
            "tokens": cleaned.split(),
            "entities": [],
        }

    # ── Tokens + Lemmas ───────────────────────────────────────────────
    tokens: list[str] = []
    lemmas: list[str] = []

    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.text)
            lemma = word.lemma if word.lemma else word.text
            lemmas.append(lemma)

    lemmatized_text = " ".join(lemmas)

    # ── Named Entities ────────────────────────────────────────────────
    entities: list[NamedEntity] = []
    for sentence in doc.sentences:
        for ent in sentence.ents:
            entities.append(
                NamedEntity(
                    text=ent.text,
                    label=ent.type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )

    return {
        "cleaned_text": cleaned,
        "lemmatized_text": lemmatized_text,
        "tokens": tokens,
        "entities": entities,
    }

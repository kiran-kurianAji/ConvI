"""
Text Pipeline — Language Detector

Uses langdetect (pure-Python) to identify the language of each conversation
turn and of the full transcript.

At Python 3.14 fasttext-wheel cannot be compiled because it requires
Microsoft C++ Build Tools, so langdetect is the drop-in replacement.
langdetect mirrors Google's language-detection library supporting 55 languages
including English (en), Hindi (hi), Malayalam (ml), and other Indian languages.
"""

from __future__ import annotations

from langdetect import detect, detect_langs, LangDetectException
from loguru import logger


# Seed ensures reproducible results across runs (langdetect uses randomness)
from langdetect import DetectorFactory
DetectorFactory.seed = 42


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the primary language of *text*.

    Returns
    -------
    (lang_code, confidence) — e.g. ("en", 0.9999)
    Falls back to ("en", 0.0) on failure.
    """
    cleaned = text.strip()
    if not cleaned:
        return "en", 0.0

    try:
        langs = detect_langs(cleaned)
        # detect_langs returns a list sorted by probability descending
        top = langs[0]
        return top.lang, round(top.prob, 4)
    except LangDetectException as exc:
        logger.warning(f"Language detection failed for text snippet: {exc}")
        return "en", 0.0


def dominant_language(texts: list[str]) -> str:
    """
    Detect language across a list of text segments (all turns combined) and
    return the most frequently detected language code.
    """
    if not texts:
        return "en"

    combined = " ".join(t for t in texts if t.strip())
    lang, _ = detect_language(combined)
    return lang

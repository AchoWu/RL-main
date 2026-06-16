"""Math-aware sentence segmenter for SLOPD Phase 0.1 validation.

Strategy:
1. Mask LaTeX environments and inline math with placeholders to protect their
   internal punctuation from being treated as sentence boundaries.
2. Mask decimal numbers (e.g. 3.14) with placeholders.
3. Split on sentence-terminating punctuation followed by whitespace OR newline.
4. Restore placeholders.
5. Merge sentences shorter than `min_chars` with their neighbor; split sentences
   longer than `max_chars` at the nearest soft boundary.

This is intentionally simple and self-contained — no external NLP libs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Sentence-terminating punctuation. We treat \n as a strong boundary too because
# CoT outputs frequently use newlines to separate reasoning steps.
_TERMINATORS = re.compile(r"(?<=[.!?])(?=\s)|\n+")

# LaTeX patterns to protect (order matters: longer first).
_LATEX_PATTERNS = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\$[^$\n]+?\$"),
    re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.DOTALL),
]

# Decimal numbers: 3.14, 0.5, 12.345
_DECIMAL = re.compile(r"\b\d+\.\d+\b")

# Common abbreviations whose period should NOT terminate a sentence.
_ABBREVS = {"e.g.", "i.e.", "etc.", "vs.", "Mr.", "Dr.", "Fig.", "Eq.", "Sec.",
            "St.", "approx.", "no.", "No."}


@dataclass
class SegmentStats:
    """Diagnostic info for one segmented trajectory."""
    n_sentences: int
    sentence_char_lens: list[int]
    sentence_token_lens: list[int]  # rough token estimate (whitespace-split)
    too_short: int   # < min_chars after merging attempt
    too_long: int    # > max_chars after splitting attempt
    raw_text: str
    sentences: list[str]


def _mask(text: str, patterns: list[re.Pattern], tag: str) -> tuple[str, list[str]]:
    """Replace pattern matches with placeholders. Returns masked text + originals."""
    originals: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(originals)
        originals.append(m.group(0))
        return f"<<{tag}_{idx}>>"

    for pat in patterns:
        text = pat.sub(_replace, text)
    return text, originals


def _mask_decimals(text: str) -> tuple[str, list[str]]:
    originals: list[str] = []

    def _replace(m: re.Match) -> str:
        idx = len(originals)
        originals.append(m.group(0))
        return f"<<DEC_{idx}>>"

    text = _DECIMAL.sub(_replace, text)
    return text, originals


def _unmask(text: str, latex_originals: list[str], decimal_originals: list[str]) -> str:
    for i, original in enumerate(latex_originals):
        text = text.replace(f"<<LTX_{i}>>", original)
    for i, original in enumerate(decimal_originals):
        text = text.replace(f"<<DEC_{i}>>", original)
    return text


def _abbreviation_safe_split(text: str) -> list[str]:
    """Split on terminators, then merge fragments that look like abbreviations."""
    raw_pieces = _TERMINATORS.split(text)
    pieces: list[str] = []
    buf = ""
    for p in raw_pieces:
        if p is None:
            continue
        candidate = (buf + " " + p).strip() if buf else p.strip()
        # If candidate ends in a known abbreviation, keep accumulating.
        ends_abbrev = any(candidate.endswith(a) for a in _ABBREVS)
        if ends_abbrev:
            buf = candidate
        else:
            if candidate:
                pieces.append(candidate)
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


def segment(
    text: str,
    min_chars: int = 10,
    max_chars: int = 600,
) -> SegmentStats:
    """Segment a CoT-style text into sentences, math-aware.

    Args:
        text: Raw model output (may contain LaTeX, decimals, multi-line CoT).
        min_chars: Sentences shorter than this are merged with their neighbor.
        max_chars: Sentences longer than this are split at soft boundaries
            (commas, semicolons) if available.

    Returns:
        SegmentStats with sentence list and diagnostics.
    """
    if not text or not text.strip():
        return SegmentStats(
            n_sentences=0, sentence_char_lens=[], sentence_token_lens=[],
            too_short=0, too_long=0, raw_text=text, sentences=[],
        )

    # 1. Mask LaTeX
    masked, latex_origs = _mask(text, _LATEX_PATTERNS, "LTX")
    # 2. Mask decimals
    masked, decimal_origs = _mask_decimals(masked)
    # 3. Split
    pieces = _abbreviation_safe_split(masked)
    # 4. Unmask
    pieces = [_unmask(p, latex_origs, decimal_origs) for p in pieces]
    # 5. Strip + drop empty
    pieces = [p.strip() for p in pieces if p and p.strip()]

    # 6. Merge too-short
    merged: list[str] = []
    for p in pieces:
        if merged and len(p) < min_chars:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)

    # 7. Split too-long at soft boundaries
    final: list[str] = []
    for p in merged:
        if len(p) <= max_chars:
            final.append(p)
            continue
        # Try to split at ", " or "; " near the middle
        chunks = _split_long(p, max_chars)
        final.extend(chunks)

    char_lens = [len(s) for s in final]
    token_lens = [len(s.split()) for s in final]
    too_short = sum(1 for L in char_lens if L < min_chars)
    too_long = sum(1 for L in char_lens if L > max_chars)

    return SegmentStats(
        n_sentences=len(final),
        sentence_char_lens=char_lens,
        sentence_token_lens=token_lens,
        too_short=too_short,
        too_long=too_long,
        raw_text=text,
        sentences=final,
    )


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split a too-long sentence at soft boundaries (',' or ';')."""
    chunks: list[str] = []
    current = text
    while len(current) > max_chars:
        # Find the latest ", " or "; " before max_chars
        cut = -1
        for sep in ["; ", ", "]:
            idx = current.rfind(sep, 0, max_chars)
            if idx > cut:
                cut = idx + len(sep)
        if cut <= 0:
            # No soft boundary — hard cut at max_chars
            cut = max_chars
        chunks.append(current[:cut].strip())
        current = current[cut:].strip()
    if current:
        chunks.append(current)
    return chunks


# --------------- Self-tests ---------------

def _self_test() -> None:
    cases = [
        # Basic
        ("Hello. How are you?", 2),
        # Decimal protection
        ("Pi is 3.14. That is approximately three.", 2),
        # LaTeX inline protection
        (r"We have $a.b$ and end. Next sentence.", 2),
        # LaTeX block protection
        (r"\[ x = 1. y = 2. \] So that is fine.", 2),
        # Newline as boundary
        ("First step.\nSecond step.\nThird.", 3),
        # Abbreviation handling
        ("We use e.g. cars. Then trucks.", 2),
        # CoT-like
        ("Let me think. 12 × 13 = 156. The answer is 156.", 3),
    ]
    for text, expected in cases:
        result = segment(text)
        ok = "OK" if result.n_sentences == expected else "FAIL"
        print(f"[{ok}] expected={expected} got={result.n_sentences} :: {text!r}")
        if result.n_sentences != expected:
            for s in result.sentences:
                print(f"    -> {s!r}")


if __name__ == "__main__":
    _self_test()

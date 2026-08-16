"""
Numeric density scoring -- shared by adaptive chunking (Part A) and
failure-aware detection (Part B).

Built directly from a real, diagnosed failure: a chunk containing beam
size (4), length penalty (0.6), a citation number [38] appearing twice,
and an input-length offset (+50) all close together caused the reader
to extract the wrong number, even though retrieval and reranking both
correctly identified the chunk as relevant. Isolated testing confirmed
the reader gets the right answer instantly on the same sentence alone,
with no surrounding numeric noise -- so density of nearby numbers is
the actual variable at fault, not chunk relevance or reader capability
in general.
"""

import re

# Matches integers, decimals, and citation-style brackets like [38].
# Deliberately broad -- catches numbers regardless of context, since
# what matters here is raw density, not semantic meaning of each number.
_NUMBER_PATTERN = re.compile(r"\[\d+\]|\d+\.\d+|\d+")


def count_numbers(text: str) -> int:
    """Count how many distinct numeric tokens appear in a piece of text."""
    return len(_NUMBER_PATTERN.findall(text))


def numeric_density(text: str) -> float:
    """
    Numbers per 100 characters. Normalized by length so density is
    comparable across chunks of different sizes.
    """
    if not text.strip():
        return 0.0
    return count_numbers(text) / (len(text) / 100)


# Threshold picked from the diagnosed failure case: the problematic
# beam-search chunk had a density around 2.0 numbers per 100 chars.
# Chunks with the correct, cleanly-extractable answers (e.g. the
# "h = 8 parallel attention layers" chunk) sat noticeably lower.
# This is a starting point based on one confirmed case, not a
# statistically tuned value -- worth revisiting once evaluated against
# a larger before/after eval set.
HIGH_DENSITY_THRESHOLD = 1.2
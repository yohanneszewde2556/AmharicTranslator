"""
Amharic Text Normalization & Punctuation Isolation Engine.

Normalizes:
1. Amharic Homophones (ሀ/ሐ/ኀ, ሠ/ሰ, ዐ/አ, ፀ/ጸ)
2. Common informal spelling/verb variations (e.g. ሀለው -> ሃለሁ)
3. Punctuation isolation & spacing (e.g., ?, !, 。, ፡, ፡, .)
"""

import re

def normalize_amharic_text(text: str) -> str:
    """
    Standardizes Amharic homophones, normalizes informal verb endings,
    and isolates punctuation marks to prevent subword tokenizer fragmentation.

    Args:
        text (str): Raw input Amharic text

    Returns:
        str: Cleaned and normalized Amharic text
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Homophone character standardization
    # ሐ, ኀ -> ሀ
    text = re.sub(r'[ሐኀ]', 'ሀ', text)
    # ሑ, ኁ -> ሁ
    text = re.sub(r'[ሑኁ]', 'ሁ', text)
    # ሒ, ኺ -> ሂ
    text = re.sub(r'[ሒኺ]', 'ሂ', text)
    # ሓ, ኻ -> ሃ
    text = re.sub(r'[ሓኻ]', 'ሃ', text)
    # ሔ, ኼ -> ሄ
    text = re.sub(r'[ሔኼ]', 'ሄ', text)
    # ሕ, ኅ -> ህ
    text = re.sub(r'[ሕኅ]', 'ህ', text)
    # ሖ, ኆ -> ሆ
    text = re.sub(r'[ሖኆ]', 'ሆ', text)

    # ሠ -> ሰ
    text = re.sub(r'ሠ', 'ሰ', text)
    text = re.sub(r'ሡ', 'ሱ', text)
    text = re.sub(r'ሢ', 'ሲ', text)
    text = re.sub(r'ሣ', 'ሳ', text)
    text = re.sub(r'ሤ', 'ሴ', text)
    text = re.sub(r'ሥ', 'ስ', text)
    text = re.sub(r'ሦ', 'ሶ', text)

    # ዐ -> አ
    text = re.sub(r'ዐ', 'አ', text)
    text = re.sub(r'ዑ', 'ኡ', text)
    text = re.sub(r'ዒ', 'ኢ', text)
    text = re.sub(r'ዓ', 'ኣ', text)
    text = re.sub(r'ዔ', 'ኤ', text)
    text = re.sub(r'ዕ', 'እ', text)
    text = re.sub(r'ዖ', 'ኦ', text)

    # ፀ -> ጸ
    text = re.sub(r'ፀ', 'ጸ', text)
    text = re.sub(r'ፁ', 'ጹ', text)
    text = re.sub(r'ፂ', 'ጺ', text)
    text = re.sub(r'ፃ', 'ጻ', text)
    text = re.sub(r'ፄ', 'ጼ', text)
    text = re.sub(r'ፅ', 'ጽ', text)
    text = re.sub(r'ፆ', 'ጾ', text)

    # Common informal suffix variations
    text = re.sub(r'ሀለው$', 'ሃለሁ', text)
    text = re.sub(r'ሀለው\s', 'ሃለሁ ', text)
    text = re.sub(r'ሀለሁ$', 'ሃለሁ', text)
    text = re.sub(r'ሀለሁ\s', 'ሃለሁ ', text)

    # 2. Punctuation isolation & spacing
    # Insert space around question marks (?), exclamation marks (!), periods (.), commas (,), Ge'ez four-dots (።), colon (፡)
    text = re.sub(r'([?!።፣;:.,])', r' \1 ', text)

    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def strip_trailing_punctuation(text: str) -> str:
    """
    Strips trailing sentence boundary punctuation for invariant root encoding.
    """
    if not text:
        return ""
    return re.sub(r'[?!።፣;:.,\s]+$', '', text).strip()


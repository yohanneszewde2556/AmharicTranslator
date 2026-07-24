"""
Amharic Text Normalization & Punctuation Isolation Engine.

Normalizes:
1. Amharic Homophones (ሀ/ሐ/ኀ, ሠ/ሰ, ዐ/አ, ፀ/ጸ)
2. Common informal spelling/verb variations (e.g. ሀለው -> ሃለሁ)
3. Punctuation isolation & spacing (e.g., ?, !, 。, ፡, ፡, .)
4. Interrogative question mark auto-completion for question words (እንዴት, ማን, ምን, etc.)
"""

import re

# List of common Amharic interrogative question words
QUESTION_WORDS = [
    'እንዴት', 'ማን', 'ምን', 'ምንድን', 'ለምን', 'መቼ', 'የት', 'ስንት', 'እንደምን'
]

def normalize_amharic_text(text: str) -> str:
    """
    Standardizes Amharic homophones, normalizes informal verb endings,
    isolates punctuation marks, and auto-completes interrogative context for question words.

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
    text = re.sub(r'[ሑኁ]', 'ሁ', text)
    text = re.sub(r'[ሒኺ]', 'ሂ', text)
    text = re.sub(r'[ሓኻ]', 'ሃ', text)
    text = re.sub(r'[ሔኼ]', 'ሄ', text)
    text = re.sub(r'[ሕኅ]', 'ህ', text)
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
    text = re.sub(r'([?!።፣;:.,])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Interrogative Question Mark Auto-Completion
    # If the input contains a question word or already has '?', ensure it ends with isolated '?'
    has_question_word = any(qw in text for qw in QUESTION_WORDS)
    if (has_question_word or '?' in text) and not text.endswith('?'):
        text = text.rstrip('.!። ') + ' ?'

    return text.strip()

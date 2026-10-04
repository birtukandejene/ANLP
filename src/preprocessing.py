# src/preprocessing.py
import re
import string
from typing import List, Dict, Optional

# Amharic Unicode ranges
AMHARIC_RANGE = r'[\u1200-\u137F]'
AMHARIC_PUNCTUATION = r'[፡።፣፤፥፦፧፨]'

# Common Amharic abbreviations and their expansions
ABBREVIATION_MAP = {
    'ት/ቤት': 'ትምህርት ቤት',
    'ወ/ሮ': 'ወይዘሮ',
    'አ/ር': 'አቶ',
    'ዶ/ር': 'ዶክተር',
    'ፕ/ር': 'ፕሮፌሰር',
    'ኢ/ር': 'ኢንጂነር',
    'ሰ/ኞ': 'ሰኞ',
    'ማ/ም': 'ማክሰኞ',
    'ረ/ዕ': 'ረቡዕ',
    'ሐ/ሙ': 'ሐሙስ',
    'ዓ/ር': 'ዓርብ',
    'ቅ/ም': 'ቅዳሜ',
    'እ/ሁ': 'እሁድ',
    'ሰ/ት': 'ሰኞ እስከ ዓርብ',
    'ክ/ል': 'ክልል',
    'ጠ/ሚ': 'ጠቅላይ ሚኒስትር',
    'ም/ጉ': 'ምክር ቤት',
    'ፌ/ዲ/ር': 'ፌደራል',
    'መ/ም': 'መምህር',
    'ፕ/ት': 'ፕሮቶኮል',
    'ኮ/ሚ': 'ኮሚሽን',
    'ሚ/ን': 'ሚኒስቴር',
    'ዩ/ኒ': 'ዩኒቨርሲቲ',
    'ቢ/ሮ': 'ቢሮ',
    'ክ/ፍ': 'ክፍል',
}

# Amharic normalization mapping
NORMALIZATION_MAP = {
    'ሀ': 'ሀ', 'ሁ': 'ሁ', 'ሂ': 'ሂ', 'ሃ': 'ሃ', 'ሄ': 'ሄ', 'ህ': 'ህ', 'ሆ': 'ሆ',
    'ሐ': 'ሀ', 'ሑ': 'ሁ', 'ሒ': 'ሂ', 'ሓ': 'ሃ', 'ሔ': 'ሄ', 'ሕ': 'ህ', 'ሖ': 'ሆ',
    'ኀ': 'ሀ', 'ኁ': 'ሁ', 'ኂ': 'ሂ', 'ኃ': 'ሃ', 'ኄ': 'ሄ', 'ኅ': 'ህ', 'ኆ': 'ሆ',
    'ሠ': 'ሰ', 'ሡ': 'ሱ', 'ሢ': 'ሲ', 'ሣ': 'ሳ', 'ሤ': 'ሴ', 'ሥ': 'ስ', 'ሦ': 'ሶ',
    'ሰ': 'ሰ', 'ሱ': 'ሱ', 'ሲ': 'ሲ', 'ሳ': 'ሳ', 'ሴ': 'ሴ', 'ስ': 'ስ', 'ሶ': 'ሶ',
    'ሸ': 'ሸ', 'ሹ': 'ሹ', 'ሺ': 'ሺ', 'ሻ': 'ሻ', 'ሼ': 'ሼ', 'ሽ': 'ሽ', 'ሾ': 'ሾ',
    'ቀ': 'ቀ', 'ቁ': 'ቁ', 'ቂ': 'ቂ', 'ቃ': 'ቃ', 'ቄ': 'ቄ', 'ቅ': 'ቅ', 'ቆ': 'ቆ',
    'በ': 'በ', 'ቡ': 'ቡ', 'ቢ': 'ቢ', 'ባ': 'ባ', 'ቤ': 'ቤ', 'ብ': 'ብ', 'ቦ': 'ቦ',
    'ቨ': 'ቨ', 'ቩ': 'ቩ', 'ቪ': 'ቪ', 'ቫ': 'ቫ', 'ቬ': 'ቬ', 'ቭ': 'ቭ', 'ቮ': 'ቮ',
    'ተ': 'ተ', 'ቱ': 'ቱ', 'ቲ': 'ቲ', 'ታ': 'ታ', 'ቴ': 'ቴ', 'ት': 'ት', 'ቶ': 'ቶ',
    'ቸ': 'ቸ', 'ቹ': 'ቹ', 'ቺ': 'ቺ', 'ቻ': 'ቻ', 'ቼ': 'ቼ', 'ች': 'ች', 'ቾ': 'ቾ',
    'ነ': 'ነ', 'ኑ': 'ኑ', 'ኒ': 'ኒ', 'ና': 'ና', 'ኔ': 'ኔ', 'ን': 'ን', 'ኖ': 'ኖ',
    'ኘ': 'ኘ', 'ኙ': 'ኙ', 'ኚ': 'ኚ', 'ኛ': 'ኛ', 'ኜ': 'ኜ', 'ኝ': 'ኝ', 'ኞ': 'ኞ',
    'አ': 'አ', 'ኡ': 'ኡ', 'ኢ': 'ኢ', 'ኣ': 'ኣ', 'ኤ': 'ኤ', 'እ': 'እ', 'ኦ': 'ኦ',
    'ከ': 'ከ', 'ኩ': 'ኩ', 'ኪ': 'ኪ', 'ካ': 'ካ', 'ኬ': 'ኬ', 'ክ': 'ክ', 'ኮ': 'ኮ',
    'ኸ': 'ኸ', 'ኹ': 'ኹ', 'ኺ': 'ኺ', 'ኻ': 'ኻ', 'ኼ': 'ኼ', 'ኽ': 'ኽ', 'ኾ': 'ኾ',
    'ወ': 'ወ', 'ዉ': 'ዉ', 'ዊ': 'ዊ', 'ዋ': 'ዋ', 'ዌ': 'ዌ', 'ው': 'ው', 'ዎ': 'ዎ',
    'ዐ': 'አ', 'ዑ': 'ኡ', 'ዒ': 'ኢ', 'ዓ': 'ኣ', 'ዔ': 'ኤ', 'ዕ': 'እ', 'ዖ': 'ኦ',
    'ዘ': 'ዘ', 'ዙ': 'ዙ', 'ዚ': 'ዚ', 'ዛ': 'ዛ', 'ዜ': 'ዜ', 'ዝ': 'ዝ', 'ዞ': 'ዞ',
    'ዠ': 'ዠ', 'ዡ': 'ዡ', 'ዢ': 'ዢ', 'ዣ': 'ዣ', 'ዤ': 'ዤ', 'ዥ': 'ዥ', 'ዦ': 'ዦ',
    'የ': 'የ', 'ዩ': 'ዩ', 'ዪ': 'ዪ', 'ያ': 'ያ', 'ዬ': 'ዬ', 'ይ': 'ይ', 'ዮ': 'ዮ',
    'ደ': 'ደ', 'ዱ': 'ዱ', 'ዲ': 'ዲ', 'ዳ': 'ዳ', 'ዴ': 'ዴ', 'ድ': 'ድ', 'ዶ': 'ዶ',
    'ጀ': 'ጀ', 'ጁ': 'ጁ', 'ጂ': 'ጂ', 'ጃ': 'ጃ', 'ጄ': 'ጄ', 'ጅ': 'ጅ', 'ጆ': 'ጆ',
    'ገ': 'ገ', 'ጉ': 'ጉ', 'ጊ': 'ጊ', 'ጋ': 'ጋ', 'ጌ': 'ጌ', 'ግ': 'ግ', 'ጎ': 'ጎ',
    'ጠ': 'ጠ', 'ጡ': 'ጡ', 'ጢ': 'ጢ', 'ጣ': 'ጣ', 'ጤ': 'ጤ', 'ጥ': 'ጥ', 'ጦ': 'ጦ',
    'ጨ': 'ጨ', 'ጩ': 'ጩ', 'ጪ': 'ጪ', 'ጫ': 'ጫ', 'ጬ': 'ጬ', 'ጭ': 'ጭ', 'ጮ': 'ጮ',
    'ጰ': 'ጰ', 'ጱ': 'ጱ', 'ጲ': 'ጲ', 'ጳ': 'ጳ', 'ጴ': 'ጴ', 'ጵ': 'ጵ', 'ጶ': 'ጶ',
    'ጸ': 'ጸ', 'ጹ': 'ጹ', 'ጺ': 'ጺ', 'ጻ': 'ጻ', 'ጼ': 'ጼ', 'ጽ': 'ጽ', 'ጾ': 'ጾ',
    'ፀ': 'ፀ', 'ፁ': 'ፁ', 'ፂ': 'ፂ', 'ፃ': 'ፃ', 'ፄ': 'ፄ', 'ፅ': 'ፅ', 'ፆ': 'ፆ',
    'ፈ': 'ፈ', 'ፉ': 'ፉ', 'ፊ': 'ፊ', 'ፋ': 'ፋ', 'ፌ': 'ፌ', 'ፍ': 'ፍ', 'ፎ': 'ፎ',
    'ፐ': 'ፐ', 'ፑ': 'ፑ', 'ፒ': 'ፒ', 'ፓ': 'ፓ', 'ፔ': 'ፔ', 'ፕ': 'ፕ', 'ፖ': 'ፖ',
}

def normalize_amharic(text: str) -> str:
    """
    Normalize Amharic characters to their standard forms.
    Converts variant characters (ሐ, ኀ, ሠ, etc.) to their standard equivalents.
    """
    if not text:
        return text
    
    # Apply character normalization
    for old_char, new_char in NORMALIZATION_MAP.items():
        text = text.replace(old_char, new_char)
    
    return text

def expand_abbreviations(text: str) -> str:
    """
    Expand common Amharic abbreviations to their full forms.
    """
    if not text:
        return text
    
    # Sort by length (longest first) to handle nested abbreviations
    sorted_abbr = sorted(ABBREVIATION_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    
    for abbr, expansion in sorted_abbr:
        # Use word boundaries to avoid partial matches
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
    
    return text

def clean_text(text: str) -> str:
    """
    Enhanced cleaning: trim whitespace, remove unwanted punctuation, 
    clean extra spaces, and handle noise.
    """
    if not text:
        return text
    
    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove unwanted punctuation (keep Amharic punctuation, periods, commas)
    # Keep: Amharic punctuation (፡።፣፤፥፦፧፨), periods, commas, question marks, exclamation
    # Remove: other punctuation like @, #, $, %, ^, &, *, (, ), etc.
    unwanted_punct = r'[@"#$%^&*()_+=\[\]{}|\\:;<>/?`~]'

    def replace_punctuation(match):
        start, end = match.span()
        if start > 0 and end < len(text) and text[start - 1].isalpha() and text[end].isalpha():
            return match.group()
        return ' '

    text = re.sub(unwanted_punct + '+', replace_punctuation, text)
    
    # 4. Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 5. Remove standalone punctuation marks (like a period or comma by itself)
    text = re.sub(r'\s+[.,!?;:፡።፣፤፥፦፧፨]+\s+', ' ', text)
    
    # 6. Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using Amharic sentence boundaries.
    Returns sentences as a list.
    """
    if not text:
        return []
    
    # Amharic sentence boundaries: ። (full stop), ፤ (colon), ፥ (semicolon), ፡ (comma)
    # Also consider English punctuation for mixed texts: . ! ?
    sentence_boundaries = r'[።፡፤፥፦፧፨.!?]'
    
    # Split on sentence boundaries
    sentences = re.split(sentence_boundaries, text)
    
    # Clean up empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def preprocess_amharic_text(
    text: str,
    normalize: bool = True,
    expand_abbr: bool = True,
    clean: bool = True,
    segment: bool = False
) -> str:
    """
    Simplified preprocessing pipeline for Amharic text.
    
    Steps:
    1. Normalize characters (ሐ → ሀ, ኀ → ሀ, ሠ → ሰ, etc.)
    2. Expand abbreviations (ት/ቤት → ትምህርት ቤት)
    3. Clean text (trim, remove unwanted punctuation, extra spaces)
    4. Segment sentences (optional - each on new line)
    
    Args:
        text: Raw Amharic text
        normalize: Apply character normalization
        expand_abbr: Expand abbreviations
        clean: Apply text cleaning (trim, remove unwanted punctuation)
        segment: Segment into sentences (each on new line)
    
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    processed = text
    
    # Step 1: Normalize characters
    if normalize:
        processed = normalize_amharic(processed)
    
    # Step 2: Expand abbreviations
    if expand_abbr:
        processed = expand_abbreviations(processed)
    
    # Step 3: Clean text (includes trimming and punctuation removal)
    if clean:
        processed = clean_text(processed)
    
    # Step 4: Segment sentences (each on new line)
    if segment:
        sentences = segment_sentences(processed)
        # Join with newline for display
        processed = '\n'.join(sentences)
    
    return processed

def get_text_statistics(text: str) -> Dict:
    """
    Get statistics about the text.
    """
    if not text:
        return {}
    
    # Count Amharic characters
    amharic_chars = re.findall(AMHARIC_RANGE, text)
    
    # Count sentences (based on newlines or Amharic punctuation)
    if '\n' in text:
        sentences = [s for s in text.split('\n') if s.strip()]
    else:
        sentences = segment_sentences(text)
    
    return {
        "total_chars": len(text),
        "total_words": len(text.split()),
        "amharic_chars": len(amharic_chars),
        "amharic_ratio": len(amharic_chars) / len(text) if len(text) > 0 else 0,
        "unique_words": len(set(text.split())),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in text.split()) / len(text.split()) if len(text.split()) > 0 else 0
    }
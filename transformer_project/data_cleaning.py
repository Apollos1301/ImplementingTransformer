import re
from typing import Optional


WHITELIST = "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"


def clean_text(text: str) -> str:
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = ''.join(c for c in text if c in WHITELIST)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_valid_pair(src: str, tgt: str, min_len: int = 5, max_len: int = 64, max_ratio: float = 3.0) -> bool:
    src_words = src.split()
    tgt_words = tgt.split()
    
    if len(src_words) < min_len or len(src_words) > max_len:
        return False
    if len(tgt_words) < min_len or len(tgt_words) > max_len:
        return False
    
    ratio = max(len(src_words), len(tgt_words)) / max(min(len(src_words), len(tgt_words)), 1)
    if ratio > max_ratio:
        return False
    
    return True


def clean_dataset(dataset, src_key: str = 'de', tgt_key: str = 'en', 
                  min_len: int = 5, max_len: int = 64, max_ratio: float = 3.0) -> list[dict]:
    cleaned = []
    for item in dataset:
        translation = item['translation']
        src = clean_text(translation[src_key])
        tgt = clean_text(translation[tgt_key])
        
        if is_valid_pair(src, tgt, min_len, max_len, max_ratio):
            cleaned.append({src_key: src, tgt_key: tgt})
    
    return cleaned

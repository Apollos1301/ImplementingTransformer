from collections import defaultdict
import re


class BPETokenizer:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
    
    def _get_word_frequencies(self, corpus: list[str]) -> dict[tuple, int]:
        word_freqs = defaultdict(int)
        for sentence in corpus:
            words = sentence.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word:
                    word_freqs[word] += 1
        return word_freqs
    
    def _get_base_vocab(self, word_freqs: dict) -> set:
        chars = set()
        for word in word_freqs.keys():
            for char in word:
                chars.add(char)
        return chars
    
    def _split_words(self, word_freqs: dict) -> dict[tuple, int]:
        splits = {}
        for word, freq in word_freqs.items():
            splits[tuple(word)] = freq
        return splits
    
    def _get_pair_frequencies(self, splits: dict[tuple, int]) -> dict[tuple, int]:
        pair_freqs = defaultdict(int)
        for word_tokens, freq in splits.items():
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, splits: dict[tuple, int], pair: tuple) -> dict[tuple, int]:
        new_splits = {}
        merged_token = pair[0] + pair[1]
        for word_tokens, freq in splits.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_splits[tuple(new_tokens)] = freq
        return new_splits
    
    def train(self, corpus: list[str]):
        word_freqs = self._get_word_frequencies(corpus)
        base_vocab = self._get_base_vocab(word_freqs)
        self.vocab = {char: idx for idx, char in enumerate(sorted(base_vocab))}
        splits = self._split_words(word_freqs)
        
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_frequencies(splits)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self._merge_pair(splits, best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = len(self.vocab)
        
        return self.vocab, self.merges
    
    def tokenize(self, text: str) -> list[str]:
        words = text.lower().split()
        tokens = []
        for word in words:
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue
            word_tokens = list(word)
            for pair, merged in self.merges.items():
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = word_tokens[:i] + [merged] + word_tokens[i + 2:]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens
    
    def encode(self, text: str) -> list[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(token, -1) for token in tokens]
    
    def decode(self, ids: list[int]) -> str:
        id_to_token = {v: k for k, v in self.vocab.items()}
        return ''.join([id_to_token.get(idx, '') for idx in ids])

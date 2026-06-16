import torch
from torch.utils.data import Dataset
import re
import jieba
from collections import Counter

def clean_text(text):
    # Remove URL, symbols, keep Chinese, English, digits
    text = re.sub(r"https?://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*", "", text)
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text)
    return text

def tokenize(text):
    return list(jieba.cut(clean_text(text)))

class Vocabulary:
    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build_vocab(self, tokenized_texts):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)
        
        # Filter by min frequency
        words = [word for word, count in counter.items() if count >= self.min_freq]
        # Sort by frequency to keep top N
        words = sorted(words, key=lambda w: counter[w], reverse=True)
        words = words[:self.max_vocab_size - 2]
        
        for idx, word in enumerate(words):
            self.word2idx[word] = idx + 2
            self.idx2word[idx + 2] = word

    def transform(self, tokens, max_len=100):
        # Convert tokens to index sequence with padding/truncating
        indices = [self.word2idx.get(token, 1) for token in tokens]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

class SpamDataset(Dataset):
    def __init__(self, data_path, vocab=None, max_len=100, is_train=True):
        self.max_len = max_len
        self.texts = []
        self.labels = []
        self.tokenized_texts = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        label = int(parts[0])
                        text = parts[1]
                        self.labels.append(label)
                        self.texts.append(text)
                        self.tokenized_texts.append(tokenize(text))
                    except ValueError:
                        continue

        if is_train and vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(self.tokenized_texts)
        else:
            self.vocab = vocab

        self.input_ids = [self.vocab.transform(tokens, max_len) for tokens in self.tokenized_texts]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

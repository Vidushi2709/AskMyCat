import json
import random
import json, random, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# encode 
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

def collate_batch(batch, max_len = 128):  # Reduced from 256 for speed
    qs, ps, ys = zip(*batch)
    enc = tokenizer(
        list(qs),
        list(ps),
        padding = True,
        truncation = True,
        max_length= max_len, 
        return_tensors="pt"
    )
    labels = torch.tensor(ys, dtype=torch.float32)
    return enc, labels

class TransformerScorer(nn.Module):
    def __init__(self, encoder, hidden = 256):
        super().__init__()
        self.encoder = encoder 
        dim = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(dim*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, enc_ips):
        out = self.encoder(**enc_ips).last_hidden_state
        # Mean pooling over tokens
        mask = enc_ips["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [B, H]

        # Split back into q/p representations: we encoded pairs, so duplicate pooled
        # Alternatively, encode separately. Here we encode concatenated pair: use pooled as joint repr.
        joint = pooled
        # If you prefer separate encodes, run encoder twice (for q and p) and concat. This is faster (single pass).
        x = torch.cat([joint, joint], dim=1)  # simple pass-through for joint repr
        return self.classifier(x).squeeze(1)
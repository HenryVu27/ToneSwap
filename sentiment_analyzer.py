import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Data, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
from typing import List, Dict, Tuple

class SentimentClassifier(nn.Module):
    def __init__(self, num_emo):
        super().__init__()
        config = AutoConfig.from_pretrained('distilroberta-base')
        config.update({"output_hidden_states":False})
        self.encoder = AutoModel.from_pretrained('distilroberta-base', config=config)

        # simple linear classifier
        self.classifier = nn.Linaer(768, num_emo)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids = input, attention_mask = mask)
        output = output[:, 0, :] # get emotion (i.e. cls token, which is a 768-dim vector)
        return self.classifier(output)

# JIT-complied scoring for faster inference
@torch.jit.script
class Scorer:
    def __init__(self, num_emo: int):
        self.num_emo = num_emo
    
    def compute_scores(self, logits: torch.Tensor) -> torch.tensor:
        return F.softmax(logits, dim=-1)

class FastEmotionAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # simple distilroberta tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        # emotion categories:
        self.emotions = {
            'joy': ['happiness', 'delight'],
            'sadness': ['grief', 'sorrow'],
            'anger': ['fury', 'rage'],
            'fear': ['anxiety', 'terror'],
            'surprise': ['amazement', 'shock'],
            'love': ['affection', 'passion'],
            'nostalgia': ['reminiscence', 'longing'],
            'hope': ['optimism', 'anticipation']
        }
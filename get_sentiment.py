
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

# Analyzer that takes in a sentence, tokenize it, then transfer to SentimentClassifier to get emotion
class SentimentAnalyzer:
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
        self.scorer = Scorer(len(self.emotions))

        self.model = SentimentClassifier(len(self.emotions))
        self.model.to(self.device)
        self.model.eval()
        self.model = torch.jit.script(self.model) # JIT compile for faster inference

        self.max_length = 128

    @torch.no_grad() 
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        # tokenize with truncation and fixed length for consistent inference time
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device)
        }

    @torch.no_grad()  # Disable gradient computation
    def analyze(self, text: str) -> Dict:
        inputs = self.preprocess_text(text)
        # inference
        logits = self.model(**inputs)
        probabilities = self.scorer.compute_scores(logits)
        emotion_scores = {
            emotion: prob.item() 
            for emotion, prob in zip(self.emotions.keys(), probabilities[0])
        }
        
        # get top emotions
        top_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'dominant_emotions': top_emotions[:3],
            'all_emotions': emotion_scores
        }
def analyze_emotions(text: str) -> Tuple[str, Dict]:
    """Fast inference helper function"""
    analyzer = SentimentAnalyzer()
    return analyzer.analyze(text)

# Example usage and benchmarking
if __name__ == "__main__":
    import time
    sample_text = """
    As I gazed at the old photographs, memories of childhood summers 
    flooded back, filling me with a bittersweet longing for those 
    carefree days, yet also grateful for the beautiful moments we shared.
    """
    
    # Warm up
    analyzer = SentimentAnalyzer()
    analyzer.analyze(sample_text)
    # Benchmark inference speed
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        analysis = analyzer.analyze(sample_text)
    
    avg_time = (time.time() - start_time) / num_iterations
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    
    # Print results
    print("\nTop Emotions:")
    for emotion, score in analysis['dominant_emotions']:
        print(f"{emotion}: {score:.3f}")
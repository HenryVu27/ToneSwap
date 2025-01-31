import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Data, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from typing import List, Dict, Tuple

class SentimentClassifier(nn.Module):
    def __init__(self, n):
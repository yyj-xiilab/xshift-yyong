# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """Cosine Classification: 더 나은 generalization을 위한 cosine similarity 기반 분류기"""
    def __init__(self, dim=384, num_classes=47, temperature=0.1):
        super(CosineClassifier, self).__init__()
        self.temperature = temperature
        self.weight = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        # Normalize input and weight
        x_norm = F.normalize(x, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        logits = torch.mm(x_norm, weight_norm.t()) / self.temperature
        return logits


class LinearClassifier(nn.Module):
    """Linear Classification: 기본 선형 분류기"""
    def __init__(self, dim=384, num_classes=47):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

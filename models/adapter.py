# coding=utf-8
import math
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation): 파라미터 효율적인 fine-tuning"""
    def __init__(self, in_dim, out_dim, rank=16, alpha=32, dropout=0.1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


class CLSAdapter(nn.Module):
    """CLS Token Adapter: CLS token을 domain-specific representation으로 변환"""
    def __init__(self, dim=384, hidden_dim=128, dropout=0.1):
        super(CLSAdapter, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, cls_token):
        """CLS token을 domain-specific representation으로 변환"""
        residual = cls_token
        adapted = self.adapter(cls_token)
        return self.layer_norm(residual + adapted)

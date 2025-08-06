# coding=utf-8
import torch
import torch.nn.functional as F


def contrastive_loss(proj_s, proj_t, temperature=0.1):
    """Contrastive loss for domain adaptation"""
    # Normalize projections
    proj_s = F.normalize(proj_s, dim=1)
    proj_t = F.normalize(proj_t, dim=1)
    
    # Similarity matrix
    sim_matrix = torch.mm(proj_s, proj_t.t()) / temperature
    
    # Positive pairs (same class, different domain)
    # For simplicity, we use all pairs as positive
    labels = torch.arange(proj_s.size(0)).to(proj_s.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

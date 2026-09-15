import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeBoundaryLoss(nn.Module):
    def __init__(self, margin_delta=0.3, lambda_boundary=0.5):
        super().__init__()
        self.delta = margin_delta
        self.lambda_b = lambda_boundary
        
    def forward(self, z_real_norm, z_text_norm, z_syn_fault, z_text_fault):
        sim_normal = F.cosine_similarity(z_real_norm, z_text_norm)
        sim_fault = F.cosine_similarity(z_syn_fault, z_text_norm)
        loss_contr = 1.0 - torch.mean(sim_normal)
        loss_boundary = torch.mean(F.relu(self.delta - (sim_normal - sim_fault)))
        return loss_contr + self.lambda_b * loss_boundary

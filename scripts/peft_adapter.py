import torch
import torch.nn as nn

class BottleneckAdapter(nn.Module):
    def __init__(self, in_dim=512, mid_dim=64):
        super().__init__()
        self.down = nn.Linear(in_dim, mid_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(mid_dim, in_dim)
        
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class DynamicGatedFusion(nn.Module):
    def __init__(self, audio_dim=512, timbral_dim=7):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(audio_dim + timbral_dim, 1),
            nn.Sigmoid()
        )
        self.proj_t = nn.Linear(timbral_dim, audio_dim)
        
    def forward(self, z_a, x_t):
        concat = torch.cat([z_a, x_t], dim=-1)
        g = self.gate(concat)
        z_t_proj = self.proj_t(x_t)
        return g * z_a + (1 - g) * z_t_proj

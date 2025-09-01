import torch
from torch import nn
from .lm_backbone import ContinuousTransformerAudioLMBackbone

_SCALE_OFFSET = 1e-2

class LaplaceLanguageModel(nn.Module):
    def __init__(self, dim, lm_config):
        super().__init__()
        self.optimizer_cfg = lm_config.get("optimizer", {})
        backbone_cfg = lm_config.get("backbone", {})
        self.backbone = ContinuousTransformerAudioLMBackbone(embed_dim=dim, **backbone_cfg)
        self.proj = nn.Linear(self.backbone.embed_dim, dim * 2)  # μ 和 log b

    def forward(self, latents):
        B,C,T = latents.shape
        x = latents.permute(0,2,1)
        h = self.backbone.model(x)        # [B,T,H]
        p = self.proj(h)                  # [B,T,2C]
        p = p.view(B, T, C, 2).permute(0,2,1,3)
        mu, log_b = p[...,0], p[...,1]

        # adjust the scale by subtracting the offset before log
        b_raw = log_b.exp()
        b_adj = (b_raw - _SCALE_OFFSET).clamp(min=1e-6)
        log_b = torch.log(b_adj)
        
        return mu, log_b
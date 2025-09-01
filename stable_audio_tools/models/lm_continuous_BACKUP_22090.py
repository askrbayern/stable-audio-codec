import torch
from torch import nn
from .lm_backbone import ContinuousTransformerAudioLMBackbone

<<<<<<< HEAD
_SCALE_OFFSET = -2
=======
_SCALE_OFFSET = -4
>>>>>>> a7bab4856c6596597471bb8bf417860ed176ad21

class LaplaceLanguageModel(nn.Module):
    def __init__(self, dim, lm_config):
        super().__init__()
        self.optimizer_cfg = lm_config.get("optimizer", {})
        backbone_cfg = lm_config.get("backbone", {})
        self.backbone = ContinuousTransformerAudioLMBackbone(embed_dim=dim, **backbone_cfg)
        self.proj = nn.Linear(self.backbone.embed_dim, dim * 2)  # μ and b

    def forward(self, latents):
        B,C,T = latents.shape
        x = latents.permute(0,2,1)
        h = self.backbone.model(x)        # [B,T,H]
        p = self.proj(h)                  # [B,T,2C]
        p = p.view(B, T, C, 2).permute(0,2,1,3)
        mu, log_b = p[...,0], p[...,1]

        # move scale offset outside of log, output b directly
        b = (log_b + _SCALE_OFFSET).exp()
<<<<<<< HEAD
        # _SCALE_OFFSET can be checked by running 
        # hyperparameter that needs to be changed
        # try several different values and it should be IMMEDIATELY obvious
        # 1e-4 is too small (basically like turning it off)
=======
>>>>>>> a7bab4856c6596597471bb8bf417860ed176ad21
        
        return mu, b
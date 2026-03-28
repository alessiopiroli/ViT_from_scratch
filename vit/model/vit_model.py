import torch.nn as nn
import torch

class ImagePatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = self.cfg.IMG.patch_size
        self.n_patches = (self.cfg.IMG.img_size // self.patch_size) ** 2
        self.lin_in = (self.patch_size ** 2) * 3
        self.lin_out = self.cfg.MODEL.latent_size

        self.unfold = nn.Unfold(kernel_size=self.cfg.IMG.patch_size, stride=self.cfg.IMG.patch_size)
        self.linear_proj = nn.Linear(self.lin_in, self.lin_out)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.lin_out))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.lin_out))

    def forward(self, x):
        x = self.unfold(x).transpose(-1, -2)
        x = self.linear_proj(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=-2)
        x = x + self.pos_embedding
        return x
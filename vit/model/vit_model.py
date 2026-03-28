import torch.nn as nn

class ImagePatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.unfold = nn.Unfold(kernel_size=self.cfg.IMG.patch_size, stride=self.cfg.IMG.patch_size)

    def forward(self, x):
        return self.unfold(x).transpose(-1, -2)
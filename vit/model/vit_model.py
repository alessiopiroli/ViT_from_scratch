import torch.nn as nn
import torch

class ImagePatcher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = self.cfg.IMG.patch_size
        self.n_patches = (self.cfg.IMG.img_size // self.patch_size) ** 2
        self.lin_in = (self.patch_size ** 2) * 3
        self.lin_out = self.cfg.MODEL.IMG_PATCHER.latent_dim

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
    

class AttentionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = self.cfg.MODEL.IMG_PATCHER.latent_dim
        self.out_dim = self.latent_dim // self.cfg.MODEL.ViT_ENCODER.n_heads

        self.linear_query = nn.Linear(self.latent_dim, self.out_dim)
        self.linear_key = nn.Linear(self.latent_dim, self.out_dim)
        self.linear_value = nn.Linear(self.latent_dim, self.out_dim)
    
    def forward(self, x):
        Q = self.linear_query(x)
        K = self.linear_key(x)
        V = self.linear_value(x)

        attn = (Q @ torch.transpose(K, dim0=-2, dim1=-1)) / (self.out_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = attn @ V

        return attn
    

class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = self.cfg.MODEL.ViT_ENCODER.n_heads
        self.latent_dim = self.cfg.MODEL.IMG_PATCHER.latent_dim

        self.layer_norm = nn.LayerNorm(self.latent_dim)
        self.attn_heads = nn.ModuleList([AttentionHead(self.cfg) for _ in range(self.n_heads)])
        self.attn_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim*4),
            nn.GELU(),
            nn.Linear(4*self.latent_dim, self.latent_dim)
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        attn_out = torch.cat([head(x_norm) for head in self.attn_heads], dim=-1)
        x = self.attn_proj(attn_out) + x
        x_norm = self.layer_norm(x)
        x = self.mlp(x_norm) + x
        
        return x
    

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = self.cfg.MODEL.IMG_PATCHER.latent_dim
        self.n_encoders = self.cfg.MODEL.ViT.n_encoders

        self.image_patcher = ImagePatcher(self.cfg)
        self.encoders = nn.ModuleList([ViTEncoder(self.cfg) for _ in range(self.n_encoders)])
        self.head = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.cfg.MODEL.HEAD.n_classes)
        )

    def forward(self, x):
        x = self.image_patcher(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x[:, 0, :]
        x = self.head(x)

        return x
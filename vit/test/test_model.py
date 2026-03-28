import pytest
import torch
from easydict import EasyDict as edict
from vit.model.vit_model import ImagePatcher, HeadAttention, ViTEncoder
from vit.utils.misc import load_config

@pytest.fixture
def config():
    return load_config("vit/config/vit_config.yml")

@pytest.mark.parametrize("bs, n_ch, h, w", [(4, 3, 224, 224)])
def test_img_patcher(config, bs, n_ch, h, w):
    img_size = config.IMG.img_size
    patch_size = config.IMG.patch_size
    n_patches = (img_size // patch_size) ** 2
    x = torch.randn(bs, n_ch, h, w)
    out = ImagePatcher(config)(x)
    assert out.shape == (bs, n_patches + 1, config.MODEL.IMG_PATCHER.latent_dim)

@pytest.mark.parametrize("bs, n_vec, l_dim", [(4, 197, 512)])
def test_head_attn(config, bs, n_vec, l_dim):
    x = torch.randn(bs, n_vec, l_dim)
    out = HeadAttention(config)(x)
    assert out.shape == (bs, n_vec, l_dim // config.MODEL.ViT_ENCODER.n_heads)

@pytest.mark.parametrize("bs, n_vec, l_dim", [(4, 197, 512)])
def test_vit_enc(config, bs, n_vec, l_dim):
    x = torch.randn(bs, n_vec, l_dim)
    out = ViTEncoder(config)(x)
    assert out.shape == x.shape
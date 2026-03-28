import pytest
import torch

from vit.model.vit_model import AttentionHead, ImagePatcher, ViT, ViTEncoder
from vit.utils.misc import load_config


@pytest.fixture
def cfg():
    return load_config("vit/config/vit_config.yml")


@pytest.mark.parametrize("bs, n_ch, h, w", [(4, 3, 224, 224)])
def test_img_patcher(cfg, bs, n_ch, h, w):
    img_size = cfg.IMG.img_size
    patch_size = cfg.IMG.patch_size
    n_patches = (img_size // patch_size) ** 2
    x = torch.randn(bs, n_ch, h, w)
    out = ImagePatcher(cfg)(x)
    assert out.shape == (bs, n_patches + 1, cfg.MODEL.IMG_PATCHER.latent_dim)


@pytest.mark.parametrize("bs, n_vec, l_dim", [(4, 197, 512)])
def test_head_attn(cfg, bs, n_vec, l_dim):
    x = torch.randn(bs, n_vec, l_dim)
    out = AttentionHead(cfg)(x)
    assert out.shape == (bs, n_vec, l_dim // cfg.MODEL.ViT_ENCODER.n_heads)


@pytest.mark.parametrize("bs, n_vec, l_dim", [(4, 197, 512)])
def test_vit_enc(cfg, bs, n_vec, l_dim):
    x = torch.randn(bs, n_vec, l_dim)
    out = ViTEncoder(cfg)(x)
    assert out.shape == x.shape


@pytest.mark.parametrize("bs, n_ch, h, w", [(4, 3, 224, 224)])
def test_vit(cfg, bs, n_ch, h, w):
    n_cls = cfg.MODEL.HEAD.n_classes
    x = torch.randn(bs, n_ch, h, w)
    out = ViT(cfg)(x)
    assert out.shape == (bs, n_cls)

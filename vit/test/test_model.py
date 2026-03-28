import pytest
import torch
from easydict import EasyDict as edict
from vit.model.vit_model import ImagePatcher
from vit.utils.misc import load_config

@pytest.fixture
def config():
    return load_config("vit/config/vit_config.yml")

@pytest.mark.parametrize("bs, n_ch, h, w", [(4, 3, 224, 224)])
def test_img_patcher(config, bs, n_ch, h, w):
    img_size = config.IMG.img_size
    patch_size = config.IMG.patch_size
    x = torch.randn(bs, n_ch, h, w)
    out = ImagePatcher(config)(x)
    assert out.shape == (bs, (img_size // patch_size)**2, (patch_size**2)*n_ch)
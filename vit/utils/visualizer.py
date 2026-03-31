import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms

from vit.dataset.cifrar10_dataset import IDX_TO_CLASS, CIFAR10Dataset
from vit.model.vit_model import ViT, ViTAttnExtract
from vit.utils.misc import get_device, setup_logging


class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vis_size = self.cfg.VISUALIZATION.vis_size
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self.media_dir = self.cfg.LOGGING.media_dir
        self.device = get_device(self.logger)
        self._build_dataloader()

    def _build_dataloader(self):
        self.dataset = CIFAR10Dataset(self.cfg, split="test")
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True)
        self.logger.info("Built dataloader")

    def _load_model(self):
        self.model = ViT(self.cfg)
        state_dict = torch.load(str(self.cfg.VISUALIZATION.model_path))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        self.logger.info("Loaded model")
        self.attn_extract = ViTAttnExtract(self.model)
        self.attn_extract.eval()
        self.attn_extract.to(self.device)
        self.logger.info("Built attention extractor")

    def test_model(self):
        self._load_model()
        for image, label in self.dataloader:
            image, label = image.to(self.device), label.to(self.device)
            pred_idx = self.model(image).squeeze(0).argmax(dim=0).item()
            pred = IDX_TO_CLASS[pred_idx]
            gt = IDX_TO_CLASS[label.item()]

            original = self._denormalize(image.squeeze(0))
            original = transforms.functional.to_pil_image(original)
            original = original.resize((self.vis_size, self.vis_size))
            attn_map = self._get_attention_map(image)
            overlay = self._overlay_attention(original, attn_map)
            self._vis_image(original, overlay, pred, gt)

    def _denormalize(self, img):
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=img.device).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010], device=img.device).view(3, 1, 1)
        img = img * std + mean
        return torch.clamp(img, 0, 1)

    def _vis_image(self, img, overlay, pred, gt):
        gap = 20
        header_height = 80
        text_size = 24

        total_width = self.vis_size * 2 + gap
        total_height = self.vis_size + header_height
        canvas = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))

        canvas.paste(img, (0, header_height))
        canvas.paste(overlay, (self.vis_size + gap, header_height))

        draw = ImageDraw.Draw(canvas)
        color = (80, 200, 80) if pred == gt else (220, 60, 60)

        try:
            font = ImageFont.truetype("/Library/Fonts/SF-Pro-Display-Bold.otf", text_size)
        except IOError:
            font = ImageFont.load_default()

        left_x = self.vis_size // 2
        right_x = self.vis_size + gap + self.vis_size // 2
        text_y = header_height // 2

        draw.text((left_x, text_y), f"Pred: {pred}", fill=color, font=font, anchor="mm")
        draw.text((right_x, text_y), f"GT: {gt}", fill=color, font=font, anchor="mm")

        os.makedirs(self.media_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = os.path.join(self.media_dir, f"{pred}_vs_{gt}_{timestamp}.png")
        canvas.save(save_path)

    def _get_attention_map(self, image):
        with torch.no_grad():
            attn = self.attn_extract(image)
        attn_map = attn[0].mean(dim=0)
        attn_map = attn_map[0, 1:]
        n = self.cfg.IMG.img_size // self.cfg.IMG.patch_size
        attn_map = attn_map.reshape(1, n, n)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map.detach().cpu()

    def _overlay_attention(self, original, attn_map):
        img_array = np.array(original)
        h, w = img_array.shape[:2]
        attn_rescaled = (
            F.interpolate(attn_map.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze().numpy()
        )
        cmap = plt.get_cmap("viridis")
        heatmap = cmap(attn_rescaled)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)
        alpha = 0.6
        blended = (img_array * (1 - alpha) + heatmap * alpha).astype(np.uint8)
        return Image.fromarray(blended)

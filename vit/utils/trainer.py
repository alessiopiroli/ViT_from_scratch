import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from vit.dataset.cifrar10_dataset import CIFAR10Dataset
from vit.dataset.oxford_pets_dataset import OxfordPetsDataset
from vit.dataset.tiny_imagenet_dataset import TinyImageNetDataset
from vit.model.vit_model import ViT
from vit.utils.misc import get_device, log_epoch, setup_logging


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_epochs = self.cfg.TRAINING.n_epochs
        self.batch_size = int(self.cfg.TRAINING.batch_size)
        self.lr = float(self.cfg.TRAINING.lr)
        self.logger, self.writer, self.experiment_dir = setup_logging(self.cfg)
        self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = self._build_scheduler()
        self.loss_fn = nn.CrossEntropyLoss()
        self._build_dataloaders()

    def _build_scheduler(self):
        decay = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.n_epochs)
        return decay

    def _build_dataloaders(self):
        if self.cfg.DATA.dataset == "oxford_pets":
            self.train_dataset = OxfordPetsDataset(self.cfg, split="trainval")
            self.val_dataset = OxfordPetsDataset(self.cfg, split="test")
        elif self.cfg.DATA.dataset == "tiny_imagenet":
            self.train_dataset = TinyImageNetDataset(self.cfg, split="train")
            self.val_dataset = TinyImageNetDataset(self.cfg, split="val")
        elif self.cfg.DATA.dataset == "cifrar10":
            self.train_dataset = CIFAR10Dataset(self.cfg, split="train")
            self.val_dataset = CIFAR10Dataset(self.cfg, split="test")

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.logger.info("Built training loader")

        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.logger.info("Built test loader")

    def _build_model(self):
        self.device = get_device(self.logger)
        self.model = ViT(self.cfg)
        self.model = self.model.to(self.device)
        self.logger.info("Built model")

    def train(self):
        for epoch in range(self.n_epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, accuracy = self.validate_one_epoch(epoch)
            self.scheduler.step()

            log_epoch(
                self.logger,
                self.writer,
                epoch,
                self.n_epochs,
                train_loss,
                val_loss,
                accuracy,
                self.experiment_dir,
                self.model,
            )

        self.logger.info("Training finished")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        self.logger.info(f"Training epoch {epoch+1}")
        for image, label in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            image, label = image.to(self.device), label.to(self.device)
            pred = self.model(image)
            loss = self.loss_fn(pred, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_one_epoch(self, epoch):
        self.model.eval()
        n_corr_preds = 0
        n_samples = 0
        total_loss = 0.0
        avg_loss = 0.0

        self.logger.info(f"Validate epoch {epoch+1}")
        with torch.no_grad():
            for image, label in tqdm(self.val_loader):
                image, label = image.to(self.device), label.to(self.device)
                pred = self.model(image)
                loss = self.loss_fn(pred, label)
                total_loss += loss.item()

                n_samples += label.shape[0]
                n_corr_preds += (pred.argmax(dim=1) == label).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = n_corr_preds / n_samples

        return avg_loss, accuracy

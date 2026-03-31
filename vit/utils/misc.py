import logging
import os
import time

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return edict(config)


def setup_logging(config):
    base_dir = config.LOGGING.logging_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_dir, f"exp_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    log_file = os.path.join(experiment_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    tb_dir = os.path.join(experiment_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    logger.info(f"Experiment logs saved to: {experiment_dir}")

    return logger, writer, experiment_dir


def log_epoch(logger, writer, epoch, n_epochs, train_loss, val_loss, accuracy, experiment_dir, model):
    logger.info(
        f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, accuracy: {accuracy*100:.2f}"
    )

    writer.add_scalars("loss", {"train": train_loss, "validation": val_loss}, epoch + 1)
    writer.add_scalar("validation/accuracy", accuracy * 100, epoch)

    save_path = os.path.join(experiment_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)


def get_device(logger):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

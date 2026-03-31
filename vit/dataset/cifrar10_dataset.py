import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split

        base_path = self.cfg.DATA.root_dir

        if "cifar-10-batches-py" not in base_path and os.path.exists(os.path.join(base_path, "cifar-10-batches-py")):
            self.data_dir = os.path.join(base_path, "cifar-10-batches-py")
        else:
            self.data_dir = base_path

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.cfg.IMG.img_size, self.cfg.IMG.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )

        self.data = []
        self.labels = []

        file_names = [f"data_batch_{i}" for i in range(1, 6)] if self.split == "train" else ["test_batch"]

        for file_name in file_names:
            file_path = os.path.join(self.data_dir, file_name)

            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.labels.extend(entry["labels"] if "labels" in entry else entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


IDX_TO_CLASS = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

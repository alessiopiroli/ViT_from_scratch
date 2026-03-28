import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class OxfordPetsDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.transform = transforms.Compose(
            [transforms.Resize((self.cfg.IMG.height, self.cfg.IMG.width)), transforms.ToTensor()]
        )

        self.images_dir = os.path.join(self.cfg.DATA.root_dir, "images")
        self.split_file = os.path.join(self.cfg.DATA.root_dir, "annotations", f"{self.split}.txt")
        self.samples = []

        with open(self.split_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                image_name = parts[0]
                class_id = int(parts[1]) - 1
                self.samples.append((image_name, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name, label = self.samples[idx]
        image_path = os.path.join(self.images_dir, f"{image_name}.jpg")
        image = self.transform(Image.open(image_path).convert("RGB"))
        return image, label


ID_TO_CLASS = {
    0: "Abyssinian",
    1: "American Bulldog",
    2: "American Pit Bull Terrier",
    3: "Basset Hound",
    4: "Beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "Boxer",
    9: "British Shorthair",
    10: "Chihuahua",
    11: "Egyptian Mau",
    12: "English Cocker Spaniel",
    13: "English Setter",
    14: "German Shorthaired",
    15: "Great Pyrenees",
    16: "Havanese",
    17: "Japanese Chin",
    18: "Keeshond",
    19: "Leonberger",
    20: "Maine Coon",
    21: "Miniature Pinscher",
    22: "Newfoundland",
    23: "Persian",
    24: "Pomeranian",
    25: "Pug",
    26: "Ragdoll",
    27: "Russian Blue",
    28: "Saint Bernard",
    29: "Samoyed",
    30: "Scottish Terrier",
    31: "Shiba Inu",
    32: "Siamese",
    33: "Sphynx",
    34: "Staffordshire Bull Terrier",
    35: "Wheaten Terrier",
    36: "Yorkshire Terrier",
}

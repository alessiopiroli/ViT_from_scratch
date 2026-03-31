import argparse

from vit.utils.misc import load_config
from vit.utils.trainer import Trainer


def main(args):
    cfg = load_config(args.cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, default="vit/config/vit_config.yaml", help="cfg path")
    args = parser.parse_args()
    main(args)

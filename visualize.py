import argparse

from vit.utils.misc import load_config
from vit.utils.visualizer import Visualizer


def main(args):
    config = load_config(args.config)
    visualizer = Visualizer(config)
    visualizer.test_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default="vit/config/vit_config.yaml", help="Config path")
    args = parser.parse_args()
    main(args)

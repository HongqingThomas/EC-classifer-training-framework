import argparse
import random

import numpy as np
import torch

from src.utils import config
from src.train import train_utils
from src.test import TestUtils



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(42)

    parser = argparse.ArgumentParser(description='Arguments for running the RGB IR-Fusion.')
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    trainer = train_utils(cfg)
    trainer.setup()
    # trainer.train()

    tester = TestUtils(cfg, trainer)
    tester.setup()
    # tester.test()

    tester.EC_validate_setup()
    tester.EC_classifier()
    # tester.validate_test()
    # tester.EC_validate_setup()
    tester.EC_classifier_validate()


if __name__ == '__main__':
    main()



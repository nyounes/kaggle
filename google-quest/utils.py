import os
import yaml
import random
import numpy as np
import tensorflow as tf
import torch


def load_config(config_path):
    with open('config.yaml') as file:
        config = yaml.load(file)
    return config


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)

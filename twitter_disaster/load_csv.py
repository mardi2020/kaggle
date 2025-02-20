import os
import pandas as pd
from paramters import TRAIN_PICKLE_PATH, TEST_PICKLE_PATH


def load():
    if is_already_embedded():
        return pd.read_pickle(TRAIN_PICKLE_PATH), pd.read_pickle(TEST_PICKLE_PATH)
    return pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')


def is_already_embedded():
    return os.path.exists(TRAIN_PICKLE_PATH) and os.path.exists(TEST_PICKLE_PATH)
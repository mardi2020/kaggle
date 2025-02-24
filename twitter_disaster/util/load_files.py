import os
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent


def load(flag: Literal['train', 'test'] = 'train'):
    if is_already_embedded(flag):
        columns = ['attention_mask', 'token_ids'] + (['target'] if flag == 'train' else ['id'])
        return pd.read_pickle(get_latest_file(flag))[columns]
    if flag == 'train':
        return pd.read_csv(BASE_DIR / 'data/input/train.csv',
                           usecols=['text', 'target'],
                           dtype={'text': str, 'target': np.int64})
    return pd.read_csv(BASE_DIR / 'data/input/test.csv',
                       usecols=['text', 'id'],
                       dtype={'text': str, 'target': np.int64}
                       )


def load_submission():
    return (pd.read_csv(get_latest_file('submission')),
            pd.read_csv(BASE_DIR / 'data/input/sample_submission.csv'))


def is_already_embedded(flag: Literal['train', 'test'] = 'train'):
    pkl_path = get_latest_file(flag)
    return os.path.exists(pkl_path)


def get_latest_file(prefix: Literal['train', 'test', 'model', 'submission']) -> str:
    DIR = 'data/output'
    if prefix == 'model':
        DIR = 'models'
        extension = 'pth'
        prefix = 'disaster_classifier'
    elif prefix == 'submission':
        extension = 'csv'
        prefix = 'submission'
    else:
        TRAIN_PICKLE_PREFIX = "train_bert_embeddings"
        TEST_PICKLE_PREFIX = "test_bert_embeddings"
        extension = 'pkl'
        prefix = TRAIN_PICKLE_PREFIX if prefix == 'train' else TEST_PICKLE_PREFIX

    files = glob.glob(str(BASE_DIR / DIR / f'{prefix}*.{extension}'))
    return max(files, key=os.path.getctime) if files else 'No file'

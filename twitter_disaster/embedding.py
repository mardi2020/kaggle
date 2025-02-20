from tqdm import tqdm

from preprocess import preprocess
from paramters import *
from bert import tokenize_and_convert, bert_embedding


def start_embedding(train_df, test_df):
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    tqdm.pandas()

    train_df[TOKEN_IDS] = train_df[CLEAN_TEXT].progress_apply(tokenize_and_convert)
    test_df[TOKEN_IDS] = test_df[CLEAN_TEXT].progress_apply(tokenize_and_convert)

    train_df[EMBEDDING] = train_df[TOKEN_IDS].progress_apply(bert_embedding)
    test_df[EMBEDDING] = test_df[TOKEN_IDS].progress_apply(bert_embedding)

    train_df.to_pickle(TRAIN_PICKLE_PATH)
    test_df.to_pickle(TEST_PICKLE_PATH)

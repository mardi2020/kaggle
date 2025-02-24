from util.load_files import load, is_already_embedded
from preprocess.embedding import start_embedding
from training.training import train_model
from testing.test import predict
from twitter_disaster.testing.final import final_report
from paramters import TEST_PICKLE_PATH


if __name__ == "__main__":
    # preprocessing data and embedding words
    train_df = load()
    if not is_already_embedded('train'):
        start_embedding(train_df)
    test_df = load('test')
    if not is_already_embedded('test'):
        start_embedding(test_df, path=TEST_PICKLE_PATH)

    train_model(train_df)

    predict(test_df)

    final_report()

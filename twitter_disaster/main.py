from load_csv import load, is_already_embedded
from embedding import start_embedding
from training import train_model


if __name__ == "__main__":
    train_df, test_df = load()
    if not is_already_embedded():
        start_embedding(train_df, test_df)

    train_model(train_df)

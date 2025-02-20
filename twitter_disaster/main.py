from preprocess import preprocess
from load_csv import load
from tokenizing import tokenize_and_convert

if __name__ == "__main__":
    # data 불러오기
    train_df, test_df = load()

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    train_df['token_ids'] = train_df['clean_text'].apply(tokenize_and_convert)
    test_df['token_ids'] = test_df['clean_text'].apply(tokenize_and_convert)



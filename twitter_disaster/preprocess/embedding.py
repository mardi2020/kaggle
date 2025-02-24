from tqdm import tqdm

from twitter_disaster.preprocess.preprocess import preprocess
from twitter_disaster.paramters import (
    TOKEN_IDS,
    EMBEDDING,
    CLEAN_TEXT,
    TRAIN_PICKLE_PATH,
    ATTENTION_MASK
)
from twitter_disaster.preprocess.bert import tokenize_and_convert, bert_embedding


def process_text(text):
    token_ids, attention_mask = tokenize_and_convert(text)
    embedding = bert_embedding(token_ids, attention_mask)

    return token_ids.squeeze(0).cpu().numpy(), attention_mask.squeeze(0).cpu().numpy(), embedding


def start_embedding(df, path=TRAIN_PICKLE_PATH):
    df = preprocess(df)
    tqdm.pandas()
    results = [process_text(text) for text in tqdm(df[CLEAN_TEXT].tolist(), total=len(df))]
    token_ids_list, attention_masks_list, embeddings_list = zip(*results)
    df[TOKEN_IDS] = list(token_ids_list)
    df[ATTENTION_MASK] = list(attention_masks_list)
    df.to_pickle(path)

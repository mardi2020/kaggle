from datetime import datetime
from typing import Literal

from twitter_disaster.paramters import BASE_DIR


def save_with_date(flag: Literal['model', 'submission', 'embeddings'], prefix='train'):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"models/disaster_classifier_{timestamp}.pth"
    submission_filename = f"data/output/submission_{timestamp}.csv"
    embeddings_filename = f"data/{prefix}_bert_embeddings_{timestamp}.pkl"

    if flag == 'model':
        return BASE_DIR / model_filename
    elif flag == 'submission':
        return BASE_DIR / submission_filename
    elif flag == 'embeddings':
        return BASE_DIR / embeddings_filename

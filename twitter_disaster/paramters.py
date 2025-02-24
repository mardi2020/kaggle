# csv, df
CLEAN_TEXT = 'clean_text'
TOKEN_IDS = 'token_ids'
EMBEDDING = 'embedding'
TARGET = 'target'
ATTENTION_MASK = 'attention_mask'

# training models
MAX_EPOCH = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3

# models
INPUT_DIM = 768
HIDDEN_DIM = 512  # 256 ~ 512

# BERT
MAX_LENGTH = 128
BERT_MODEL_NAME = "bert-base-uncased"

# 파일 경로 설정
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PICKLE_PATH = BASE_DIR / "data/output/train_bert_embeddings.pkl"
TEST_PICKLE_PATH = BASE_DIR / "data/output/test_bert_embeddings.pkl"
MODEL_SAVE_PATH = BASE_DIR / "models/disaster_classifier.pth"

# MPS
DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

# csv, df
CLEAN_TEXT = 'clean_text'
TOKEN_IDS = 'token_ids'
EMBEDDING = 'embedding'
TARGET = 'target'

# training model
MAX_EPOCH = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

# model
INPUT_DIM = 768
HIDDEN_DIM = 512  # 256 ~ 512

# BERT
MAX_LENGTH = 128
BERT_MODEL_NAME = "bert-base-uncased"

# 파일 경로 설정
TRAIN_PICKLE_PATH = "data/train_bert_embeddings.pkl"
TEST_PICKLE_PATH = "data/test_bert_embeddings.pkl"
MODEL_SAVE_PATH = "models/disaster_classifier.pth"

# MPS
DEVICE = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

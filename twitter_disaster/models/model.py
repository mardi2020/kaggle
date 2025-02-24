import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import DistilBertModel

from twitter_disaster.paramters import DROPOUT_RATE, DEVICE, HIDDEN_DIM, INPUT_DIM


class DisasterDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(np.vstack(embeddings), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class DisasterClassifier(nn.Module):
    def __init__(self):
        super(DisasterClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)  # BatchNorm 추가
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask
                             ).last_hidden_state[:, 0, :]
        x = self.fc1(bert_out)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

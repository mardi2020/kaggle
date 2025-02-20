import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from paramters import DROPOUT_RATE, INPUT_DIM, HIDDEN_DIM


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
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

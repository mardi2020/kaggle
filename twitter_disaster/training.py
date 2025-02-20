from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import DisasterDataset, DisasterClassifier
from paramters import (
    EMBEDDING,
    TARGET,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MAX_EPOCH
)


def train_model(train_df):
    x_train = train_df[EMBEDDING].tolist()
    y_train = train_df[TARGET].tolist()

    train_dataset = DisasterDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DisasterClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCH):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{MAX_EPOCH}, Loss: {total_loss / len(train_loader):.4f}")

    print("✅ 모델 학습 완료")

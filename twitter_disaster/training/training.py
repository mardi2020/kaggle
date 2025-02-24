import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from twitter_disaster.util.load_files import get_latest_file
from twitter_disaster.util.save_files import save_with_date
from twitter_disaster.models.model import DisasterClassifier
from twitter_disaster.paramters import (
    TOKEN_IDS,
    ATTENTION_MASK,
    TARGET,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    MAX_EPOCH
)


def save_model(model):
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict()
    }, save_with_date('model'))


def is_already_trained():
    return False if get_latest_file('model') == 'No file' else True


def train_model(train_df):
    core(
        input_ids=np.array(train_df[TOKEN_IDS].values.tolist()),
        attention_mask=np.array(train_df[ATTENTION_MASK].values.tolist()),
        labels=train_df[TARGET].values
    )


def core(input_ids, attention_mask, labels):
    input_ids = torch.tensor(input_ids, dtype=torch.int64).to(DEVICE)
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).to(DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    print(f"🔥 input_ids[:10]: {input_ids[:10]}")
    print(f"🔥 attention_mask[:10]: {attention_mask[:10]}")

    train_dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    model = DisasterClassifier().to(DEVICE)
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.array([0, 1]),
                                         y=np.array(labels))
    loop(
        model=model,
        train_loader=DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        ),
        optimizer=optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3),
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights[1], dtype=torch.float32))
    )


def loop(model: DisasterClassifier, train_loader: DataLoader, optimizer: optim, criterion):
    len_train_loader = len(train_loader)
    for epoch in range(MAX_EPOCH):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len_train_loader
        print(f"🔹 Epoch {epoch + 1}/{MAX_EPOCH}, "
              f"Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    save_model(model)

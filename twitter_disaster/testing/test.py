import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from twitter_disaster.util.load_files import get_latest_file
from twitter_disaster.util.save_files import save_with_date
from twitter_disaster.models.model import DisasterClassifier
from twitter_disaster.paramters import (
    DEVICE,
    BATCH_SIZE,
    TOKEN_IDS,
    ATTENTION_MASK
)


class TestDataset(Dataset):
    def __init__(self, df):
        self.input_ids = torch.tensor(
            np.array(df[TOKEN_IDS].values.tolist()),
            dtype=torch.int64)
        self.attention_mask = torch.tensor(
            np.array(df[ATTENTION_MASK].values.tolist()),
            dtype=torch.int64)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]


def predict(test_df):
    predications = []
    model: DisasterClassifier = load_model()

    test_dataset = TestDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    with torch.no_grad():
        model.eval()
        for input_ids, attention_mask in test_loader:
            input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)

            output = model(input_ids, attention_mask).squeeze(-1)
            probs = torch.sigmoid(output)
            batch_predictions = (probs >= 0.5).float()
            predications.extend(batch_predictions.cpu().numpy().astype(int).tolist())

    print("🔹 Unique Predictions:", np.unique(predications, return_counts=True))
    save_result(test_df, predications)


def load_model() -> DisasterClassifier:
    checkpoint = torch.load(get_latest_file('model'))
    model = DisasterClassifier().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def save_result(test_df: pd.DataFrame, preds: list[np.ndarray]):
    submission_df = pd.DataFrame({"id": test_df["id"], "target": preds})
    submission_df.to_csv(save_with_date('submission'), index=False)

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from twitter_disaster.paramters import DEVICE

MODEL_NAME = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)


def tokenize_and_convert(text):
    if not isinstance(text, str) or text.strip() == "":
        text = "[PAD]"
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].squeeze(0).to(DEVICE, dtype=torch.long)
    attention_mask = encoding["attention_mask"].squeeze(0).to(DEVICE, dtype=torch.long)

    return input_ids, attention_mask


def bert_embedding(input_ids, attention_mask):
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    return output.logits.squeeze(0).cpu().numpy()

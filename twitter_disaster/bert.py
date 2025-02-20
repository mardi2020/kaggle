import torch
from transformers import BertTokenizer, BertModel
from paramters import DEVICE

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
model.eval()  # 평가 모드


def tokenize_and_convert(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(complete_wordpiece(tokens))
    return token_ids


def complete_wordpiece(tokens):
    result, curr_word = [], ''

    for token in tokens:
        if token.startswith('##'):
            curr_word += token[2:]
            continue
        if curr_word:
            result.append(curr_word)
        curr_word = token

    return result + ([curr_word] if curr_word else None)


def bert_embedding(token_ids):
    input_tensor = torch.tensor([token_ids], dtype=torch.int32, device=DEVICE)
    attention_mask = (input_tensor != tokenizer.pad_token_id).long().to(DEVICE)

    with torch.no_grad():  # 기울기 계산 안함 - 성능 최적화..?
        output = model(input_tensor, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state

    embedding = hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

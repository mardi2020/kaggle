from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import demoji

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))


def preprocess(df):
    if "target" in df.columns:
        df = df.dropna(subset=["target"])
    df.loc[:, 'clean_text'] = df['text'].apply(cleaning_txt)
    if 'target' in df.columns:
        df = correct_mislabeled_tweets(df)
    return df


def correct_mislabeled_tweets(df):
    indices_0 = [4415, 4400, 4399, 4403, 4397, 4396, 4394, 4414, 4393, 4392,
                 4404, 4407, 4420, 4412, 4408, 4391, 4405,
                 6840, 6834, 6837, 6841, 6816, 6828, 6831,
                 246, 270, 266, 259, 253, 251, 250, 271,
                 6119, 6122, 6123, 6131, 6160, 6166, 6167, 6172, 6212, 6221, 6230, 6091, 6108,
                 7435, 7460, 7464, 7466, 7469, 7475, 7489, 7495, 7500, 7525, 7552, 7572, 7591, 7599]
    df.loc[indices_0, 'target'] = 0
    indices_1 = [3913, 3914, 3936, 3921, 3941, 3937, 3938, 3136, 3133, 3930, 3933, 3924, 3917]
    df.loc[indices_1, 'target'] = 1
    return df


def cleaning_txt(text: str):
    lemma = WordNetLemmatizer()
    text = text.lower()
    text = re.sub(r"\n", " ", text)  # 개행 제거
    text = re.sub(r"&amp;", "and", text)  # HTML 엔티티 변환
    text = re.sub(r'http\S+|www.\S+', '', text)  # URL 제거
    text = re.sub(r"@\w+", "", text)  # 멘션 제거
    text = re.sub(r"\d+", "", text)  # 숫자 제거
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자 제거
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = demoji.replace(text, "")

    words = text.split()
    words = [
        lemma.lemmatize(word) for word in words if word.lower() not in stop_words
    ]
    return " ".join(words)

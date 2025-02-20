import re


def preprocess(df):
    df = check_null_text(df)
    df['clean_text'] = df['text'].apply(cleaning_txt)
    return df


def cleaning_txt(text: str):
    text = text.lower()  # 소문자로 변환
    text = re.sub(r"http\S+", "", text)  # URL 제거
    text = re.sub(r"@\w+", "", text)  # 멘션 제거 (@username)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # 특수문자 제거
    return text.strip()


def check_null_text(df):
    # 결측치 개수 확인
    missing_cnt = df.isnull().sum()
    if missing_cnt['text'] > 0:
        df = df.dropna(subset=['text'])  # text 없는 행 삭제
    df = df.fillna('')  # 나머지는 빈 문자열로 대체
    return df

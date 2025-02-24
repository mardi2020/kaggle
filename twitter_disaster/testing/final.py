# sample_submission 과 result 대조 확인

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from twitter_disaster.util.load_files import get_latest_file
from twitter_disaster.paramters import BASE_DIR


def final_report():
    true_labels = pd.read_csv(BASE_DIR / 'data/input/sample_submission.csv')
    predicted_labels = pd.read_csv(get_latest_file('submission'))

    true_labels = true_labels.sort_values("id").reset_index(drop=True)
    predicted_labels = predicted_labels.sort_values("id").reset_index(drop=True)

    # ✅ 정답 데이터와 예측 비교
    y_true = true_labels["target"].values
    y_pred = predicted_labels["target"].values

    # ✅ 정확도, F1 Score 등 평가
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # ✅ 출력
    print(f"🔹 Accuracy: {accuracy:.4f}")
    print(f"🔹 F1 Score: {f1:.4f}")
    print(f"🔹 Precision: {precision:.4f}")
    print(f"🔹 Recall: {recall:.4f}")

    # ✅ Confusion Matrix 출력
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ✅ 상세 리포트 출력
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

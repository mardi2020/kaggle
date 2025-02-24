### 모델 구조
```
[ 입력: BERT Embeddings (768 Dim) ]
        │
        ▼
[ FC1: 768 → 256 ] # 완전연결계층 1
        │
        ▼
[ ReLU 활성화 함수 ] # 다른 비슷한거도 테스트
        │
        ▼
[ Dropout (30%) ] # 0.2 ~ 0.5 사이
        │
        ▼
[ FC2: 256 → 1 ] # 이진 분류이므로 크기는 1
        │
        ▼
[ Sigmoid 활성화 (0과 1 사이의 확률 출력) ]
        │
        ▼
[ 최종 결과: disaster(1) 또는 no(0) ]
```

https://www.kaggle.com/competitions/nlp-getting-started/overview
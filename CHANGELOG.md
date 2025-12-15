# 얼굴 인식 시스템 변경 이력

## 프로젝트 개요

이 프로젝트는 OpenCV와 TensorFlow/Keras를 사용한 얼굴 인식 시스템입니다. 웹캠을 통해 실시간으로 얼굴을 감지하고 등록된 사용자를 인식합니다.

---

## 주요 변경 사항

### 1. 모델 아키텍처 변경: LBPH → CNN

#### 초기 상태 (LBPH 기반)
- **기술**: OpenCV의 LBPH (Local Binary Patterns Histograms) 얼굴 인식기
- **특징**: 전통적인 컴퓨터 비전 기법, 딥러닝 미사용
- **모델 저장**: `trainer/trainer.yml` (OpenCV 형식)
- **장점**: 빠른 학습, 작은 데이터셋에서도 동작
- **단점**: 정확도 제한적, 복잡한 얼굴 패턴 인식 어려움

#### 변경 후 (CNN 기반)
- **기술**: TensorFlow/Keras 기반 Convolutional Neural Network (CNN)
- **특징**: 딥러닝 기반 AI 모델
- **모델 저장**: `model/face_model.h5` (Keras 형식)
- **장점**: 높은 정확도, 복잡한 패턴 학습 가능
- **단점**: 더 많은 데이터 필요, 학습 시간 증가

---

## 상세 변경 내역

### 1단계: 모델 저장 경로 변경

**변경 파일**: `model_training.py`

- **변경 전**: `trainer/trainer.yml`
- **변경 후**: `model/face_model.yml`
- **목적**: 모델 파일 관리 구조 개선

### 2단계: TensorFlow/Keras 기반 CNN 모델로 전환

#### `model_training.py` 주요 변경사항

1. **라이브러리 추가**
   ```python
   from tensorflow import keras
   from tensorflow.keras import layers
   from sklearn.preprocessing import LabelEncoder
   from sklearn.model_selection import train_test_split
   ```

2. **CNN 모델 구조**
   ```
   Input (100x100x1 grayscale)
   ├── Conv2D(32) → MaxPooling → Dropout(0.25)
   ├── Conv2D(64) → MaxPooling → Dropout(0.25)
   ├── Conv2D(128) → MaxPooling → Dropout(0.25)
   ├── Flatten
   ├── Dense(512) → Dropout(0.5)
   ├── Dense(256) → Dropout(0.5)
   └── Output (num_classes) - Softmax
   ```

3. **데이터 전처리**
   - 이미지 크기 통일: 100x100 픽셀
   - 정규화: 0-255 → 0.0-1.0
   - Train/Validation 분할 (80:20)

4. **ID 매핑 시스템**
   - 원본 ID와 모델 인덱스 간 매핑 저장
   - `id_mapping.pkl` 파일로 저장

5. **단일 클래스 데이터셋 처리**
   - 단일 사용자 데이터셋에서 `stratify` 오류 방지
   - 단순 분할 방식으로 대체

#### `recognition.py` 주요 변경사항

1. **모델 로딩**
   - Keras 모델 로드 (`model/face_model.h5`)
   - ID 매핑 파일 로드 (`model/id_mapping.pkl`)

2. **예측 방식 변경**
   - LBPH: `recognizer.predict()` → (id, confidence)
   - CNN: `model.predict()` → 확률 분포 배열

3. **Confidence Threshold 조정**
   - 변경 전: 0.7
   - 변경 후: 0.5 (더 나은 감지를 위해)

4. **Names 배열 구조 변경**
   - 리스트 → 딕셔너리
   - 원본 ID를 키로 사용

---

## 문제 해결

### 문제: 모든 얼굴을 KimSungAh로 인식

#### 원인 분석

1. **LabelEncoder 인코딩 문제**
   - 원본 ID 1 → 인코딩된 인덱스 0
   - `names[0]`이 'None'이어야 하는데 잘못된 매핑

2. **Names 배열 인덱싱 오류**
   - 인코딩된 인덱스를 사용하여 잘못된 이름 참조

3. **단일 클래스 데이터셋**
   - 한 명의 사용자만 있어 모델 학습 제한적

#### 해결 방법

1. **직접 ID 매핑 시스템 구현**
   ```python
   unique_ids = sorted(np.unique(ids))
   id_to_index = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
   index_to_id = {idx: orig_id for orig_id, idx in id_to_index.items()}
   ```

2. **ID 매핑 파일 저장**
   - `model/id_mapping.pkl`에 원본 ID와 인덱스 매핑 저장
   - 추론 시 올바른 원본 ID로 변환

3. **Names 딕셔너리 구조**
   ```python
   names = {}
   for orig_id in unique_ids:
       if orig_id == 1:
           names[orig_id] = 'KimSungAh'
       else:
           names[orig_id] = f'User_{orig_id}'
   ```

4. **단일 클래스 처리 개선**
   - `stratify` 옵션 조건부 적용
   - 단일 클래스 경고 메시지 추가

---

## 파일 구조

```
face_recognition/
├── data_create.py          # 얼굴 데이터 수집 스크립트
├── model_training.py       # CNN 모델 훈련 스크립트
├── recognition.py          # 실시간 얼굴 인식 스크립트
├── haarcascade_frontalface_default.xml  # 얼굴 검출기
├── dataset/                # 훈련 데이터셋
│   └── User.{id}.{count}.jpg
└── model/                  # 훈련된 모델
    ├── face_model.h5       # Keras CNN 모델
    ├── id_mapping.pkl      # ID 매핑 정보
    └── face_model.yml      # (구) OpenCV 모델 (호환성)
```

---

## 사용 방법

### 1. 데이터 수집
```bash
python data_create.py 1  # 사용자 ID 1의 얼굴 데이터 수집
```

### 2. 모델 훈련
```bash
python model_training.py
```

### 3. 얼굴 인식
```bash
python recognition.py
```

---

## 기술 스택

### 필수 라이브러리
- `opencv-contrib-python`: 얼굴 검출 및 이미지 처리
- `tensorflow`: 딥러닝 모델 프레임워크
- `numpy`: 수치 연산
- `PIL (Pillow)`: 이미지 처리
- `scikit-learn`: 데이터 전처리 및 분할

### 설치 명령
```bash
pip install opencv-contrib-python tensorflow scikit-learn pillow numpy
```

---

## 모델 성능

### CNN 모델 하이퍼파라미터
- **이미지 크기**: 100x100 (grayscale)
- **배치 크기**: 32
- **에포크**: 50
- **옵티마이저**: Adam
- **손실 함수**: Sparse Categorical Crossentropy
- **정규화**: Dropout (0.25, 0.5)

### 예상 성능
- **훈련 정확도**: 데이터셋 크기에 따라 다름
- **검증 정확도**: 일반적으로 80-95% (데이터 품질에 따라)
- **추론 속도**: 실시간 처리 가능 (웹캠 기준)

---

## 향후 개선 사항

1. **데이터 증강 (Data Augmentation)**
   - 회전, 밝기 조정, 노이즈 추가 등
   - 작은 데이터셋에서 모델 성능 향상

2. **전이 학습 (Transfer Learning)**
   - 사전 훈련된 모델 (VGG, ResNet 등) 활용
   - 더 빠른 학습 및 높은 정확도

3. **다중 사용자 지원 개선**
   - 동적 사용자 추가 기능
   - 사용자별 이름 매핑 파일

4. **성능 최적화**
   - 모델 경량화 (Quantization)
   - 추론 속도 향상

5. **에러 처리 강화**
   - 더 상세한 예외 처리
   - 사용자 친화적인 에러 메시지

---

## 주의사항

1. **데이터셋 크기**
   - 최소 사용자당 20-30개 샘플 권장
   - 다양한 조명, 각도, 표정 포함

2. **단일 사용자 데이터셋**
   - 한 명의 사용자만 있으면 모델 성능 제한적
   - "unknown" 얼굴 구분 어려움

3. **Confidence Threshold**
   - 환경에 따라 조정 필요
   - 너무 높으면: 인식 실패 증가
   - 너무 낮으면: 오인식 증가

4. **모델 재훈련**
   - 새로운 사용자 추가 시 모델 재훈련 필요
   - 기존 모델 덮어쓰기 주의

---

## 변경 날짜

- **2024년**: LBPH → CNN 모델 전환
- **2024년**: ID 매핑 시스템 구현
- **2024년**: 단일 클래스 데이터셋 처리 개선

---

## 참고 자료

- [OpenCV 얼굴 인식 문서](https://docs.opencv.org/)
- [TensorFlow/Keras 문서](https://www.tensorflow.org/api_docs)
- [CNN 얼굴 인식 논문](https://arxiv.org/)

---

## 작성자

이 문서는 얼굴 인식 시스템의 변경 이력을 기록하기 위해 작성되었습니다.


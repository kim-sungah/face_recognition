# 얼굴 인식 시스템 (Face Recognition System)

CNN 기반 실시간 얼굴 인식 시스템입니다. TensorFlow/Keras를 사용하여 딥러닝 모델을 훈련하고, 웹캠을 통해 실시간으로 얼굴을 인식합니다.

## 주요 기능

- 📸 **얼굴 데이터 수집**: 웹캠을 통한 얼굴 이미지 자동 수집
- 🧠 **CNN 모델 훈련**: 딥러닝 기반 얼굴 인식 모델 학습
- 👁️ **실시간 인식**: 웹캠을 통한 실시간 얼굴 인식 및 사용자 식별
- 📊 **훈련 결과 시각화**: 손실 및 정확도 그래프 자동 생성
- 🔌 **nCube 서버 연동**: 인식 결과를 외부 서버로 전송

## 프로젝트 구조

```
face_recognition/
├── data_create.py              # 얼굴 데이터 수집 스크립트
├── model_training.py           # CNN 모델 훈련 스크립트
├── recognition.py              # 실시간 얼굴 인식 스크립트
├── unknown_data_collect.py     # Unknown 클래스 데이터 수집
├── face_requirements.txt      # Python 패키지 의존성
├── haarcascade_frontalface_default.xml  # 얼굴 감지용 Haar Cascade
├── dataset/                    # 얼굴 이미지 데이터셋
│   └── User.{id}.{number}.jpg
└── model/                      # 훈련된 모델 파일
    ├── face_model.h5          # 훈련된 모델
    ├── id_mapping.pkl         # ID 매핑 정보
    └── training_history.png   # 훈련 결과 그래프
```

## 설치 방법

### 1. 필수 요구사항

- Python 3.7 이상
- 웹캠
- Windows/Linux/macOS

### 2. 패키지 설치

```bash
pip install -r face_requirements.txt
```

또는 개별 설치:

```bash
pip install opencv-python numpy Pillow tensorflow scikit-learn matplotlib
```

### 3. Haar Cascade 파일

프로젝트에 `haarcascade_frontalface_default.xml` 파일이 포함되어 있어야 합니다. 
없는 경우 OpenCV에서 다운로드하거나 다음 경로에서 확인하세요:
- OpenCV 설치 경로: `opencv/data/haarcascades/`

## 사용 방법

### 1단계: 얼굴 데이터 수집

사용자의 얼굴 이미지를 수집합니다. 각 사용자마다 최소 50-100개의 샘플을 권장합니다.

```bash
# 사용자 ID 1로 데이터 수집
python data_create.py 1

# 사용자 ID 2로 데이터 수집
python data_create.py 2
```

**사용법:**
- 웹캠이 자동으로 실행됩니다
- 얼굴이 감지되면 자동으로 저장됩니다
- ESC 키를 누르면 종료됩니다
- 최대 100개의 샘플을 수집합니다

**파일명 형식:** `User.{id}.{number}.jpg`
- 예: `User.1.1.jpg`, `User.1.2.jpg`, ...

### 2단계: Unknown 클래스 데이터 수집 (선택사항)

알 수 없는 얼굴을 구분하기 위해 Unknown 클래스(ID=0) 데이터를 수집합니다.

```bash
python unknown_data_collect.py
# 또는
python data_create.py 0
```

**권장사항:** Unknown 클래스는 최소 50개 이상의 다양한 얼굴 샘플을 수집하세요.

### 3단계: 모델 훈련

수집한 데이터로 CNN 모델을 훈련합니다.

```bash
python model_training.py
```

**훈련 과정:**
- 데이터셋 자동 로드 및 전처리
- CNN 모델 생성 및 컴파일
- 모델 훈련 (기본: 50 epochs)
- 훈련 결과 그래프 생성 (`model/training_history.png`)
- 모델 저장 (`model/face_model.h5`)
- ID 매핑 저장 (`model/id_mapping.pkl`)

**훈련 파라미터 조정:**
- `epochs`: `model_training.py` 파일에서 수정 (기본값: 50)
- `batch_size`: 기본값 32
- `IMG_SIZE`: 이미지 크기 (기본값: 100x100)

### 4단계: 실시간 얼굴 인식

훈련된 모델을 사용하여 실시간 얼굴 인식을 수행합니다.

```bash
# 기본 실행
python recognition.py

# 카메라 ID 지정 (기본값: 0)
python recognition.py -c 1

# 화면 반전 비활성화
python recognition.py --no-flip

# 통계 정보 미표시
python recognition.py --no-stats

# 커스텀 모델 경로 지정
python recognition.py -m model/custom_model.h5
```

**키보드 컨트롤:**
- `q`: 프로그램 종료
- `s`: 현재 프레임 스크린샷 저장
- `r`: 통계 정보 표시 토글

**인식 결과:**
- 얼굴 영역에 초록색 사각형 표시
- 인식된 사용자 이름 표시
- 신뢰도(Confidence) 퍼센트 표시
- nCube 서버로 결과 전송 (서버 연결 시)

## 명령줄 옵션

### recognition.py

```
-h, --help          도움말 표시
-m, --model         모델 파일 경로 (기본값: model/face_model.h5)
-i, --id-mapping    ID 매핑 파일 경로 (기본값: model/id_mapping.pkl)
-c, --camera        카메라 ID (기본값: 0)
--no-flip           화면 좌우 반전 비활성화
--no-stats          통계 정보 미표시
```

## 모델 구조

CNN 모델은 다음과 같은 구조를 가집니다:

```
Input (100x100x1)
  ↓
Conv2D(32) + MaxPooling + Dropout(0.25)
  ↓
Conv2D(64) + MaxPooling + Dropout(0.25)
  ↓
Conv2D(128) + MaxPooling + Dropout(0.25)
  ↓
Flatten
  ↓
Dense(512) + Dropout(0.5)
  ↓
Dense(256) + Dropout(0.5)
  ↓
Output (num_classes) - Softmax
```

## nCube 서버 연동

프로그램은 nCube 서버(기본: `127.0.0.1:3105`)로 인식 결과를 전송할 수 있습니다.

**서버 연결 실패 시:**
- 프로그램은 경고 메시지를 표시하고 독립적으로 작동합니다
- 서버 없이도 얼굴 인식 기능은 정상적으로 동작합니다

**전송 형식:**
```json
{
  "ctname": "face_recognition",
  "con": "{name},{confidence_percent}"
}
```

## 파일 설명

### data_create.py
- 웹캠을 통해 얼굴 이미지를 수집
- 사용자 ID를 인자로 받아 데이터셋 생성
- 자동으로 얼굴 감지 및 저장

### model_training.py
- 데이터셋에서 이미지 로드 및 전처리
- CNN 모델 생성 및 훈련
- 훈련 결과 시각화 및 모델 저장

### recognition.py
- 훈련된 모델 로드
- 실시간 얼굴 감지 및 인식
- 결과 표시 및 서버 전송

## 문제 해결

### 웹캠이 열리지 않을 때
- 웹캠이 다른 프로그램에서 사용 중인지 확인
- 카메라 ID를 변경해보세요 (`-c 1`, `-c 2` 등)
- 관리자 권한으로 실행해보세요

### 모델 파일을 찾을 수 없을 때
- `model_training.py`를 먼저 실행하여 모델을 생성하세요
- `model/` 폴더가 존재하는지 확인하세요

### 인식 정확도가 낮을 때
- 각 사용자당 더 많은 샘플 수집 (100개 이상 권장)
- 다양한 조명 조건에서 데이터 수집
- Unknown 클래스 데이터 추가 수집
- 모델 훈련 epochs 증가
- `CONFIDENCE_THRESHOLD` 값 조정

### 메모리 부족 오류
- `batch_size`를 줄이세요 (기본값: 32 → 16)
- 이미지 크기(`IMG_SIZE`)를 줄이세요 (100 → 80)

## 성능 최적화

- **GPU 사용**: TensorFlow가 GPU를 인식하면 자동으로 사용됩니다
- **배치 크기 조정**: GPU 메모리에 맞게 `batch_size` 조정
- **이미지 크기**: 더 작은 이미지 크기로 더 빠른 처리 가능

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 참고 자료

- [OpenCV 공식 문서](https://opencv.org/)
- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras 공식 문서](https://keras.io/)

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

---

**주의사항:**
- 이 시스템은 교육 및 연구 목적으로 설계되었습니다
- 실제 보안 목적으로 사용 시 추가 검증이 필요합니다
- 개인정보 보호 규정을 준수하여 사용하세요


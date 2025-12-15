import cv2
import numpy as np
from PIL import Image
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Path for face image database
path = 'dataset'

# Image dimensions for CNN
IMG_SIZE = 100

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        try:
            # Load image
            PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
            img_numpy = np.array(PIL_img, 'uint8')
            
            # Extract ID from filename (User.1.1.jpg -> 1)
            filename = os.path.split(imagePath)[-1]
            id = int(filename.split(".")[1])
            
            # Resize image to fixed size for CNN
            if img_numpy.size > 0:
                resized_img = cv2.resize(img_numpy, (IMG_SIZE, IMG_SIZE))
                faceSamples.append(resized_img)
                ids.append(id)
        except Exception as e:
            print(f"[WARNING] Error processing {imagePath}: {e}")
            continue
    
    return faceSamples, ids

# Create CNN model
def create_model(num_classes):
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Function to visualize training history
def plot_training_history(history, save_path=None):
    """
    훈련 히스토리를 시각화합니다.
    
    Args:
        history: model.fit()의 반환값
        save_path: 그래프를 저장할 경로 (None이면 화면에만 표시)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 정확도 그래프
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] 훈련 결과 그래프가 {save_path}에 저장되었습니다.")
    
    plt.show()

print("\n [INFO] 데이터를 로딩하는 중...")
faces, ids = getImagesAndLabels(path)

if len(faces) == 0:
    print("[ERROR] 얼굴 이미지를 찾을 수 없습니다. dataset 폴더를 확인하세요.")
    exit(1)

print(f"[INFO] {len(faces)}개의 얼굴 이미지를 로드했습니다.")

# Convert to numpy arrays and normalize
faces_array = np.array(faces)
faces_array = faces_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension
faces_array = faces_array.astype('float32') / 255.0  # Normalize to [0, 1]

# Encode labels - map IDs to 0-based indices while preserving original ID mapping
unique_ids = sorted(np.unique(ids))
id_to_index = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
ids_encoded = np.array([id_to_index[id] for id in ids])
num_classes = len(unique_ids)

print(f"[INFO] {num_classes}명의 사용자를 발견했습니다.")
print(f"[INFO] 사용자 ID 매핑: {id_to_index}")

# Unknown 클래스 확인 (ID 0)
UNKNOWN_ID = 0
if UNKNOWN_ID in unique_ids:
    unknown_count = len([id for id in ids if id == UNKNOWN_ID])
    print(f"[INFO] Unknown 클래스 발견: {unknown_count}개 샘플")
    if unknown_count < 50:
        print(f"[WARNING] Unknown 샘플이 부족합니다. 최소 50개 이상 권장합니다.")
else:
    print(f"[WARNING] Unknown 클래스(ID={UNKNOWN_ID})가 없습니다.")
    print(f"[INFO] Unknown 얼굴을 구분하려면 ID {UNKNOWN_ID}로 데이터를 수집하세요.")
    print(f"[INFO] 예: python unknown_data_collect.py 또는 python data_create.py {UNKNOWN_ID}")

# Split data into train and validation sets
# Note: stratify only works with multiple classes
if num_classes > 1:
    X_train, X_val, y_train, y_val = train_test_split(
        faces_array, ids_encoded, test_size=0.2, random_state=42, stratify=ids_encoded
    )
else:
    # For single class, use simple split
    split_idx = int(len(faces_array) * 0.8)
    X_train, X_val = faces_array[:split_idx], faces_array[split_idx:]
    y_train, y_val = ids_encoded[:split_idx], ids_encoded[split_idx:]
    print("[WARNING] 단일 사용자 데이터셋입니다. 모델 성능이 제한적일 수 있습니다.")

# Create model
print("\n [INFO] CNN 모델을 생성하는 중...")
model = create_model(num_classes)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train model
print("\n [INFO] 모델 훈련을 시작합니다. 시간이 걸릴 수 있습니다...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=1
)

# Create model directory if it doesn't exist
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"[INFO] {model_dir} 디렉토리를 생성했습니다.")

# Save the model
model_path = os.path.join(model_dir, 'face_model.h5')
model.save(model_path)
print(f"\n [INFO] 모델이 {model_path}에 저장되었습니다.")

# Save ID mapping for inference
import pickle
id_mapping_path = os.path.join(model_dir, 'id_mapping.pkl')
with open(id_mapping_path, 'wb') as f:
    pickle.dump({
        'id_to_index': id_to_index,
        'index_to_id': {idx: orig_id for orig_id, idx in id_to_index.items()},
        'unique_ids': unique_ids
    }, f)
print(f"[INFO] ID 매핑이 {id_mapping_path}에 저장되었습니다.")

# Visualize training results
print("\n [INFO] 훈련 결과를 시각화하는 중...")
plot_path = os.path.join(model_dir, 'training_history.png')
plot_training_history(history, save_path=plot_path)

# Print training summary
print(f"\n [INFO] 훈련 완료!")
print(f"  - 총 샘플 수: {len(faces)}")
print(f"  - 사용자 수: {num_classes}")
print(f"  - 훈련 정확도: {history.history['accuracy'][-1]:.4f}")
print(f"  - 검증 정확도: {history.history['val_accuracy'][-1]:.4f}")
print(f"  - 최종 훈련 손실: {history.history['loss'][-1]:.4f}")
print(f"  - 최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
print("[INFO] 프로그램을 종료합니다.")

import cv2
import numpy as np
import os
import pickle
from tensorflow import keras
import socket
import json
import argparse

# nCube 서버 설정
HOST = '127.0.0.1'
PORT = 3105

# nCube 연결 (선택적)
upload_client = None
try:
    upload_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upload_client.connect((HOST, PORT))
    print(f"[OK] nCube 서버 연결 성공: {HOST}:{PORT}")
except Exception as e:
    print(f"[WARNING] nCube 서버 연결 실패: {e}")
    print("[INFO] 서버 없이 독립적으로 작동합니다.")

def send_cin(con, msg):
    """
    nCube 서버로 데이터를 전송합니다.
    
    Args:
        con: 연결 이름
        msg: 전송할 메시지
    """
    if upload_client is None:
        return
    
    try:
        cin = {'ctname': con, 'con': msg}
        msg_encoded = (json.dumps(cin) + '<EOF>')
        upload_client.sendall(msg_encoded.encode('utf-8'))
        print(f"[SEND] {msg} to {con}")
    except Exception as e:
        print(f"[ERROR] 서버 전송 실패: {e}")

def close_server_connection():
    """
    서버 연결을 닫습니다.
    """
    global upload_client
    if upload_client is not None:
        try:
            upload_client.close()
            print("[OK] 서버 연결 종료")
        except Exception as e:
            print(f"[WARNING] 서버 연결 종료 실패: {e}")
        finally:
            upload_client = None

def detect_faces(gray, face_cascade, min_size=(30, 30)):
    """
    얼굴을 감지합니다.
    
    Args:
        gray: 그레이스케일 이미지
        face_cascade: Haar Cascade 분류기
        min_size: 최소 얼굴 크기
    
    Returns:
        얼굴 좌표 리스트 [(x, y, w, h), ...]
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=min_size,
    )
    return faces

def draw_prediction(img, x, y, w, h, name, confidence_percent, font):
    """
    예측 결과를 이미지에 그립니다.
    
    Args:
        img: 이미지
        x, y, w, h: 얼굴 영역 좌표
        name: 예측된 이름
        confidence_percent: 신뢰도 (퍼센트)
        font: 폰트
    """
    # 얼굴 영역 사각형
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 이름 표시
    cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
    
    # 신뢰도 표시
    cv2.putText(img, f"  {confidence_percent}%", (x+5, y+h-5), font, 1, (255, 255, 0), 1)

def run_webcam(model, index_to_id, names, camera_id=0, flip=True, show_stats=True):
    """
    웹캠을 통해 실시간 얼굴 인식을 수행합니다.
    
    Args:
        model: 훈련된 모델
        index_to_id: 인덱스에서 원본 ID로 매핑
        names: ID에서 이름으로 매핑
        camera_id: 카메라 ID (기본값: 0)
        flip: 화면 좌우 반전 여부 (기본값: True)
        show_stats: 통계 정보 표시 여부 (기본값: True)
    """
    # 웹캠 초기화
    print(f"\n웹캠 초기화 중... (카메라 ID: {camera_id})")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다 (ID: {camera_id})")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("[OK] 웹캠 준비 완료!")
    print("\n컨트롤:")
    print("  - 'q': 종료")
    print("  - 's': 스크린샷 저장")
    print("  - 'r': 통계 정보 토글")
    print("\n영상을 시작합니다...\n")
    
    # Image size for CNN (must match training)
    IMG_SIZE = 100
    
    # Haar Cascade for face detection
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Define min window size to be recognized as a face
    minW = 0.1 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    minH = 0.1 * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Confidence threshold for CNN
    CONFIDENCE_THRESHOLD = 0.5
    
    # 통계 정보
    frame_count = 0
    faces_detected = 0
    show_stats_flag = show_stats
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            
            if not ret:
                print("[WARNING] 프레임을 읽을 수 없습니다.")
                break
            
            # 화면 좌우 반전
            if flip:
                frame = cv2.flip(frame, 1)
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 감지
            faces = detect_faces(gray, faceCascade, (int(minW), int(minH)))
            
            # 감지된 얼굴에 대해 예측 수행
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출
                face_roi = gray[y:y+h, x:x+w]
                
                # Preprocess for CNN
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_array = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                face_array = face_array.astype('float32') / 255.0
                
                # Predict using CNN
                predictions = model.predict(face_array, verbose=0)
                predicted_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_index]
                
                # Convert index back to original ID
                predicted_id = index_to_id[predicted_index]
                
                # Check if confidence is above threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Valid match found
                    if predicted_id in names:
                        name = names[predicted_id]
                    else:
                        name = f"ID_{predicted_id}"
                    confidence_percent = round(confidence * 100)
                else:
                    # Unknown face - confidence too low
                    name = "unknown"
                    confidence_percent = round(confidence * 100)
                
                # 결과 표시
                draw_prediction(frame, x, y, w, h, name, confidence_percent, font)
                
                # nCube 서버로 결과 전송
                send_cin("face_recognition", f"{name},{confidence_percent}")
                
                faces_detected += 1
            
            # 통계 정보 표시
            if show_stats_flag:
                frame_count += 1
                stats_text = f"Frames: {frame_count} | Faces detected: {faces_detected}"
                cv2.putText(frame, stats_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 프레임 표시
            cv2.imshow("Real-time Face Recognition - Press 'q' to quit", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q'):
                print("\n[OK] 사용자 요청으로 종료합니다.")
                break
            elif key == ord('s'):
                # 스크린샷 저장
                screenshot_path = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"[OK] 스크린샷 저장: {screenshot_path}")
            elif key == ord('r'):
                # 통계 정보 토글
                show_stats_flag = not show_stats_flag
                print(f"[INFO] 통계 정보 표시: {'ON' if show_stats_flag else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n[INFO] 인터럽트로 종료합니다.")
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()
        close_server_connection()
        print("\n[OK] 웹캠 종료 완료")
        print(f"[INFO] 총 처리된 프레임: {frame_count}")
        print(f"[INFO] 감지된 얼굴 수: {faces_detected}")

def main():
    """
    메인 실행 함수
    """
    parser = argparse.ArgumentParser(description='웹캠을 통한 실시간 얼굴 인식')
    parser.add_argument('-m', '--model', type=str, default='model/face_model.h5',
                       help='모델 파일 경로 (기본값: model/face_model.h5)')
    parser.add_argument('-i', '--id-mapping', type=str, default='model/id_mapping.pkl',
                       help='ID 매핑 파일 경로 (기본값: model/id_mapping.pkl)')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='카메라 ID (기본값: 0)')
    parser.add_argument('--no-flip', action='store_true',
                       help='화면 좌우 반전 비활성화')
    parser.add_argument('--no-stats', action='store_true',
                       help='통계 정보 미표시')
    
    args = parser.parse_args()
    
    # 모델 파일 확인
    print("=" * 70)
    print("실시간 얼굴 인식 - 웹캠 모드")
    print("=" * 70)
    print(f"\n모델 로딩: {args.model}")
    
    if not os.path.exists(args.model):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {args.model}")
        print("[INFO] 먼저 model_training.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    if not os.path.exists(args.id_mapping):
        print(f"[ERROR] ID 매핑 파일을 찾을 수 없습니다: {args.id_mapping}")
        print("[INFO] 먼저 model_training.py를 실행하여 모델을 학습시켜주세요.")
        return
    
    # 모델 로드
    try:
        print("[INFO] 모델을 로딩하는 중...")
        model = keras.models.load_model(args.model)
        print("[OK] 모델 로드 완료!")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return
    
    # ID 매핑 로드
    try:
        with open(args.id_mapping, 'rb') as f:
            id_mapping = pickle.load(f)
            index_to_id = id_mapping['index_to_id']
            unique_ids = id_mapping['unique_ids']
        print(f"[INFO] 로드된 사용자 ID: {unique_ids}")
    except Exception as e:
        print(f"[ERROR] ID 매핑 로드 실패: {e}")
        return
    
    # Names related to ids - dynamically create based on loaded IDs
    UNKNOWN_ID = 0
    names = {}
    for orig_id in unique_ids:
        if orig_id == UNKNOWN_ID:
            names[orig_id] = 'unknown'
        elif orig_id == 1:
            names[orig_id] = 'KimSungAh'
        else:
            names[orig_id] = f'User_{orig_id}'
    
    print(f"[INFO] Unknown 클래스 ID: {UNKNOWN_ID}")
    
    # 웹캠 실행
    run_webcam(
        model=model,
        index_to_id=index_to_id,
        names=names,
        camera_id=args.camera,
        flip=not args.no_flip,
        show_stats=not args.no_stats
    )

if __name__ == "__main__":
    main()

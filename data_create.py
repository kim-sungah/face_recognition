import cv2
import os
import sys

# dataset 폴더가 없으면 생성
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print("[INFO] Created dataset folder")

# 명령줄 인자로 사용자 ID 받기
if len(sys.argv) > 1:
    face_id = sys.argv[1]
else:
    try:
        face_id = input('\n enter user id and press <return> ==> ')
    except (EOFError, KeyboardInterrupt):
        print("\n[ERROR] 사용자 ID가 필요합니다.")
        print("사용법: python data_create.py <user_id>")
        print("예시: python data_create.py 1")
        sys.exit(1)

if not face_id:
    print("[ERROR] 사용자 ID를 입력해주세요.")
    print("사용법: python data_create.py <user_id>")
    sys.exit(1)

# 웹캠 초기화
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하세요.")
    sys.exit(1)

cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Haar Cascade 파일 확인
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"[ERROR] {cascade_path} 파일을 찾을 수 없습니다.")
    sys.exit(1)

face_detector = cv2.CascadeClassifier(cascade_path)

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
print(f"[INFO] 사용자 ID: {face_id}")
print("[INFO] ESC 키를 누르면 종료됩니다.")

# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        filename = "dataset/User." + str(face_id) + '.' + str(count) + ".jpg"
        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        print(f"[INFO] Saved: {filename} ({count}/100)")
    
    cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 100: # Take 100 face samples and stop video
        print(f"\n[INFO] {count}개의 얼굴 샘플을 수집했습니다.")
        break

# Do a bit of cleanup
print("\n [INFO] Exiting program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
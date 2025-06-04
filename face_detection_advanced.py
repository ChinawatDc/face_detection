import cv2
import datetime
import os
import threading

# ฟังก์ชันเล่นเสียงแบบไม่บล็อก (เรียกใช้แยก thread)
def play_sound_async(path):
    pass  # ไม่เล่นเสียง

# โหลด Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ตั้งค่าพารามิเตอร์ตรวจจับ
scaleFactor = 1.1
minNeighbors = 5
minSize = (30, 30)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

prev_faces_count = 0  # เก็บจำนวนใบหน้าครั้งก่อนหน้า เพื่อเปรียบเทียบ

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    # ส่งเสียงแจ้งเตือนเมื่อเจอใบหน้าใหม่ (เพิ่มขึ้น)
    if len(faces) > prev_faces_count and len(faces) > 0:
        play_sound_async('alert_sound.mp3')  # เตรียมไฟล์เสียง alert_sound.mp3 ไว้ในโฟลเดอร์เดียวกับสคริปต์
    prev_faces_count = len(faces)

    for (x, y, w, h) in faces:
        # วาดกรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ตรวจจับดวงตาภายในใบหน้า
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # ตรวจจับรอยยิ้มภายในใบหน้า
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)

    # แสดงจำนวนใบหน้าที่ตรวจจับได้
    cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงเวลาปัจจุบัน
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    frame_height = frame.shape[0]
    cv2.putText(frame, current_time, (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # แสดงผล
    cv2.imshow('Advanced Face Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # กด 'q' เพื่อออก
        break

# ปิดทุกอย่าง
cap.release()
cv2.destroyAllWindows()

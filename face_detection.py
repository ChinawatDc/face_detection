import cv2

# โหลดไฟล์ cascade สำหรับตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)  # 0 คือกล้องตัวแรกของเครื่อง

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพเป็นขาวดำ (grayscale) เพราะ cascade ต้องการภาพแบบนี้
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # วาดกรอบสี่เหลี่ยมรอบ ๆ ใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    # แสดงภาพผลลัพธ์
    cv2.imshow('Face Detection', frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()

# 🧠 face_detection

โปรเจกต์ตรวจจับใบหน้าด้วย OpenCV และ MediaPipe แบบเรียลไทม์ พร้อมฟีเจอร์ขั้นสูง เช่น ตรวจจับดวงตาและรอยยิ้ม

---

## 🔧 1. ความต้องการระบบ (System Requirements)

- Windows 10 หรือใหม่กว่า  
- **Python 3.10 หรือ 3.11** ✅ *(ไม่รองรับ Python 3.13)*  
- Visual Studio Build Tools (บางระบบอาจต้องใช้)  
- กล้อง Webcam (ในตัวหรือ USB)

---

## 🐍 2. ติดตั้ง Python 3.10 หรือ 3.11

> หากคุณมี Python 3.13 อยู่แล้ว **แนะนำให้ติดตั้ง Python 3.10 เพิ่มต่างหาก**

1. เข้าไปที่ [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. ดาวน์โหลด **Python 3.10.x หรือ 3.11.x**
3. ตอนติดตั้ง ให้ติ๊ก ✅ "Add Python to PATH"
4. คลิก Install

---

## 📦 3. สร้าง Virtual Environment (แนะนำ)

```powershell
cd D:\myCode\ai\face_detection
py -3.10 -m venv venv
.\venv\Scripts\activate
📥 4. ติดตั้งไลบรารีที่จำเป็น
```


## 📥 4. ติดตั้งไลบรารีที่จำเป็น
```bash
pip install opencv-python mediapipe
```
หาก mediapipe ติดตั้งไม่ได้ แสดงว่า Python เวอร์ชันของคุณไม่รองรับ (ต้องต่ำกว่า 3.12)

## 📄 5. ไฟล์ในโปรเจกต์
| ไฟล์                         | คำอธิบาย                                                           |
| ---------------------------- | ------------------------------------------------------------------ |
| `face_detection.py`          | ตรวจจับใบหน้าแบบง่ายด้วย OpenCV Haar Cascade                       |
| `face_detection_advanced.py` | ตรวจจับใบหน้า + ดวงตา + รอยยิ้ม แสดงจำนวนใบหน้าและเวลาแบบเรียลไทม์ |

## ▶️ 6. การใช้งานแต่ละไฟล์
#### ✅ face_detection.py
ฟีเจอร์:

ตรวจจับใบหน้าแบบเรียลไทม์

แสดงกรอบรอบใบหน้าที่ตรวจพบ

วิธีรัน:
```bash
python face_detection.py
```

#### ✅ face_detection_advanced.py
ฟีเจอร์:

ตรวจจับใบหน้า + ดวงตา + รอยยิ้ม

แสดงเวลา และจำนวนใบหน้าบนจอ

มีโค้ดแจ้งเตือนเสียง (ปิดไว้ชั่วคราว)

ไม่มีการบันทึกรูป/วิดีโอแล้ว

วิธีรัน:
```bash
python face_detection_advanced.py
```

🛑 ปุ่มลัด
q → ออกจากโปรแกรม

c → บันทึกภาพใบหน้า (ฟีเจอร์นี้ถูกลบออกแล้วในเวอร์ชันปัจจุบัน)

❗ ปัญหาที่พบบ่อย
| ปัญหา                                              | วิธีแก้                            |
| -------------------------------------------------- | ---------------------------------- |
| `ModuleNotFoundError: No module named 'mediapipe'` | ใช้ Python 3.10 หรือ 3.11 เท่านั้น |
| กล้องไม่ทำงาน                                      | ปิดโปรแกรมอื่นที่อาจกำลังใช้กล้อง  |
| เปิด `python` แล้วเข้า Microsoft Store             | ใช้คำสั่ง `py` หรือ `py -3.10` แทน |


📂 โครงสร้างโปรเจกต์
face_detection/
│
├── face_detection.py
├── face_detection_advanced.py
├── venv/                  ← Virtual environment (หลังสร้าง)
└── detected_faces/        ← โฟลเดอร์จะถูกสร้างอัตโนมัติ (หากมีฟีเจอร์บันทึกภาพเปิดไว้)

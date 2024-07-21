import cv2
import mediapipe as mp
import numpy as np
import joblib

# โหลดโมเดลที่ฝึกแล้วและ scaler
model = joblib.load('hand_gesture_model.pkl')
scaler = joblib.load('scaler.pkl')

# เริ่มต้น MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# เริ่มต้น MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# เริ่มต้นการทำงานของเว็บแคม.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # แปลงภาพจาก BGR เป็น RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # ประมวลผลภาพและตรวจจับมือ.
    results = hands.process(image)

    # วาดจุดเชื่อมโยงบนมือ.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # สกัดจุดเชื่อมโยงสำหรับการรู้จำท่าทาง.
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            landmarks = np.array(landmarks).flatten()
            
            # มาตรฐานข้อมูล
            landmarks = scaler.transform([landmarks])
            
            # พยากรณ์ท่าทาง
            gesture = model.predict(landmarks)[0]
            print(f"ท่าทางที่ตรวจจับได้: {gesture}")

            # แสดงผลลัพธ์เป็นข้อความบนหน้าจอ
            cv2.putText(image, f"detect: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # แสดงภาพ.
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


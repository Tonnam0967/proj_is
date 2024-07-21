import cv2
import mediapipe as mp
import numpy as np
import joblib

# โหลดโมเดล
model = joblib.load('hand_gesture_model.pkl')
scaler = joblib.load('scaler.pkl')

# Start MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Start MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR --> RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # tracking hands
    results = hands.process(image)

    # draw point
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # เชื่อมจุด
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)
            landmarks = np.array(landmarks).flatten()
            
            # get data csv
            landmarks = scaler.transform([landmarks])
            
            # result
            gesture = model.predict(landmarks)[0]
            print(f"ท่าทางที่ตรวจจับได้: {gesture}")

            # show result on window
            cv2.putText(image, f"detect: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # show img.
    cv2.imshow('Hand Detection', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


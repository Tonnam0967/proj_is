import cv2
import mediapipe as mp
import numpy as np
import csv

# start MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# start MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# start webcam
cap = cv2.VideoCapture(0)

# set get_data.
gestures = ['headache', 'sore_throat', 'runny_nose', 'eye_hurt']
current_gesture = gestures[0]

# csv
with open('hand_gesture_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)])
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # tracking hands
        results = hands.process(image)

        # draw
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
                
                # write to CSV.
                writer.writerow([current_gesture] + landmarks)
        
        # แสดงข้อความบนภาพเพื่อระบุท่าทางที่กำลังเก็บข้อมูล.
        cv2.putText(image, f"Recording Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "Press keys to change gestures:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        for i, gesture in enumerate(gestures):
            cv2.putText(image, f"{i+1}: {gesture}", (10, 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # แสดงภาพ.
        cv2.imshow('Hand Detection', image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_gesture = gestures[0]
        elif key == ord('2'):
            current_gesture = gestures[1]
        elif key == ord('3'):
            current_gesture = gestures[2]
        elif key == ord('4'):
            current_gesture = gestures[3]

cap.release()
cv2.destroyAllWindows()

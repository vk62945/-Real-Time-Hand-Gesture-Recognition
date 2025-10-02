import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create dataset folder
if not os.path.exists("gesture_data"):
    os.makedirs("gesture_data")

# Ask user which gesture to record
gesture_name = input("Enter gesture name (wave / handshake / approach): ").strip().lower()
file_path = f"gesture_data/{gesture_name}.csv"

# Open CSV file
csv_file = open(file_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)

cap = cv2.VideoCapture(0)
print("Recording... Press ESC to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmark_list = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save landmarks if detected
        if landmark_list:
            csv_writer.writerow(landmark_list)

    cv2.imshow("Recording Gesture Data", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"Saved data for gesture: {gesture_name} in {file_path}")

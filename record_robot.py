# record_gestures.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# --- Setup ---
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
gesture_name = input("Enter gesture name (wave/shake/approach): ")

folder_path = "gesture"
os.makedirs(folder_path, exist_ok=True)
file_path = os.path.join(folder_path, f"{gesture_name}.csv")

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    coords = (coords - min_vals) / (max_vals - min_vals + 1e-6)
    return coords.flatten()

def compute_angles(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    angles = []
    for i in range(len(coords)-1):
        for j in range(i+1, len(coords)):
            angles.append(np.linalg.norm(coords[i]-coords[j]))
    return np.array(angles)

data = []
print("Recording started... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            norm_landmarks = normalize_landmarks(hand_landmarks.landmark)
            angles = compute_angles(hand_landmarks.landmark)
            row = np.concatenate([norm_landmarks, angles])
            data.append(row)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Recording gestures", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# --- Save gesture data ---
df = pd.DataFrame(data)
df.to_csv(file_path, mode='a', header=False, index=False)
print(f"Saved {len(data)} samples to {file_path}")
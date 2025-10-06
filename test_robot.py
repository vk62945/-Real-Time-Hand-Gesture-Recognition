# robot_response.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Load model & class names ---
model = tf.keras.models.load_model("gesture_model.h5")
gestures = np.load("classes.npy")

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
pred_queue = deque(maxlen=5)

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
            X = np.concatenate([norm_landmarks, angles]).reshape(1, -1)

            pred = model.predict(X, verbose=0)
            pred_queue.append(np.argmax(pred))
            gesture_idx = max(set(pred_queue), key=pred_queue.count)
            gesture = gestures[gesture_idx]

            # Draw hand landmarks & bounding box
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)*w), int(max(x_coords)*w)
            y_min, y_max = int(min(y_coords)*h), int(max(y_coords)*h)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            # Draw arrow pointing away from palm
            wrist = hand_landmarks.landmark[0]
            index_mcp = hand_landmarks.landmark[5]
            pinky_mcp = hand_landmarks.landmark[17]
            cx = int(((index_mcp.x + pinky_mcp.x)/2) * w)
            cy = int(((index_mcp.y + pinky_mcp.y)/2) * h)
            nx = int((cx - wrist.x * w) * 0.5)
            ny = int((cy - wrist.y * h) * 0.5)
            arrow_end = (cx + nx, cy + ny)
            cv2.arrowedLine(frame, (cx, cy), arrow_end, (255, 0, 0), 3, tipLength=0.3)

            # Robot response
            if gesture == "wave":
                response = "Robot waves back 👋"
            elif gesture == "shake":
                response = "Robot shakes hand 🤝"
            elif gesture == "approach":
                response = "Robot steps back ↩️"
            else:
                response = "Idle"

            cv2.putText(frame, f"{gesture} -> {response}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Robot Response", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

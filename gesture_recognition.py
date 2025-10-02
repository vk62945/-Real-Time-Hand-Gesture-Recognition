import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import csv
import os

# -------------------------------
# Settings
# -------------------------------
SAVE_DATA = True            # Set True to save landmarks while testing
GESTURE_NAME = "wave"       # Change dynamically per gesture session
DATA_FOLDER = "gesture_data"  

# Create folder if not exists
if SAVE_DATA and not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

data_file_path = os.path.join(DATA_FOLDER, f"{GESTURE_NAME}.csv")
csv_file = None
csv_writer = None
if SAVE_DATA:
    csv_file = open(data_file_path, mode="a", newline="")
    csv_writer = csv.writer(csv_file)

# -------------------------------
# Load model
# -------------------------------
try:
    model = tf.keras.models.load_model("gesture_model.h5")
    print("✅ Model loaded successfully")
except:
    print("⚠️ Model not found. Running only data recording.")

# Load gesture labels
GESTURES = []
if os.path.exists("gesture_labels.txt"):
    with open("gesture_labels.txt") as f:
        for line in f:
            GESTURES.append(line.strip().split(":")[1])

# -------------------------------
# Mediapipe setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# -------------------------------
# Open Camera
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera. Close other apps using webcam.")
    exit()
print("✅ Camera opened. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # -------------------------------
    # Hand detection
    # -------------------------------
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

            # Save data dynamically
            if SAVE_DATA and csv_writer:
                csv_writer.writerow(landmark_list)

            # Predict gesture if model exists
            if 'model' in globals() and GESTURES:
                landmark_array = np.array(landmark_list).reshape(1, -1)
                prediction = model.predict(landmark_array, verbose=0)
                class_id = np.argmax(prediction)
                gesture_name = GESTURES[class_id]

                # Display prediction
                cv2.putText(frame, gesture_name, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                # Debug
                print("🤖 Predicted gesture:", gesture_name)

    cv2.imshow("Gesture Live + Record", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        print("🔴 Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if SAVE_DATA and csv_file:
    csv_file.close()
    print(f"✅ Landmarks saved to {data_file_path}")

# ✋-Real-Time-Hand-Gesture-Recognition

#Features
1. Record gestures via webcam → CSV (record_gestures.py)
2. Train gesture recognition model → MLP (train_model.py)
3. Real-time recognition & robot response (robot_response.py)
4. Visual feedback: hand skeleton, bounding box, arrow pointing away from palm

#Scripts
1. record_gestures.py – Record gestures and save as CSV.
2. train_model.py – Train neural network on recorded gestures.
3. robot_response.py – Real-time gesture recognition and robot response display.

#Supported Gestures
1. wave → Robot waves back 👋
2. shake → Robot shakes hand 🤝
3. approach → Robot steps back ↩️

#Algorithms
1. MediaPipe Hands → hand landmark detection
2. Feature extraction → normalized coordinates + pairwise distances
3. MLP neural network → gesture classification
4. Deque smoothing → stable real-time predictions

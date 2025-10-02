# -Real-Time-Hand-Gesture-Recognition

**Real-Time Hand Gesture Recognition using MediaPipe, TensorFlow, and OpenCV**  
---
## 📖 About  
This project lets you record hand movements, train a model, and recognize gestures in real time using your webcam.  
It uses **MediaPipe** to detect hand landmarks, **TensorFlow** to learn gesture patterns, and **OpenCV** to show live predictions.  

Complete pipeline: **Data Collection → Model Training → Real-Time Prediction**  
---
## 📂 Project Structure  
### 1️⃣ Data Collection – `record_gesture_data.py`  
- Captures hand landmarks (x, y coordinates).  
- Saves them into CSV files inside `gesture_data/`.  
- Example: `wave.csv`, `handshake.csv`, `approach.csv`.  
### 2️⃣ Model Training – `train_gesture_model.py`  
- Loads all gesture CSV files.  
- Prepares dataset and one-hot encodes labels.  
- Builds and trains a simple neural network.  
- Saves:  
  - `gesture_model.h5` → trained model.  
  - `gesture_labels.txt` → gesture label mapping.  
### 3️⃣ Real-Time Recognition – `gesture_recognition.py`  
- Opens webcam and detects hand in real time.  
- Predicts gestures using trained model.  
- Displays hand skeleton + predicted gesture on screen.  
- Optionally records new gesture data.  
---
## 🛠️ Tech Stack  
- Python 3.x  
- OpenCV (video capture + display)  
- MediaPipe (hand tracking)  
- TensorFlow / Keras (gesture classification)  
- NumPy, Pandas (data processing)  
---


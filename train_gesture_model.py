import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import glob
import os

# Load all CSV files
data = []
labels = []
gesture_map = {}
label_id = 0

for file in glob.glob("gesture_data/*.csv"):
    gesture_name = os.path.splitext(os.path.basename(file))[0]
    gesture_map[label_id] = gesture_name
    
    df = pd.read_csv(file, header=None)
    X = df.values
    y = np.full((X.shape[0],), label_id)
    
    data.append(X)
    labels.append(y)
    label_id += 1

# Prepare dataset
X = np.vstack(data)
y = np.hstack(labels)

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=len(gesture_map))

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(gesture_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

# Save model + label map
model.save("gesture_model.h5")

# Save gesture map
with open("gesture_labels.txt", "w") as f:
    for k, v in gesture_map.items():
        f.write(f"{k}:{v}\n")

print("Model training complete! Saved as gesture_model.h5 and gesture_labels.txt")

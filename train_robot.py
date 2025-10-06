# train_model.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

folder_path = "gesture"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

data_list, label_list = [], []

print("Loading gesture data...")
for file in files:
    label = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(folder_path, file), header=None)
    data_list.append(df.values)
    label_list += [label] * len(df)

X = np.vstack(data_list).astype(float)
y = np.array(label_list)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
num_classes = len(encoder.classes_)
print("Loaded classes:", encoder.classes_)

# Normalize data
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-6)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Build improved model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, shuffle=True)
model.save("gesture_model.h5")

# Save encoder classes for later
np.save("classes.npy", encoder.classes_)

print("Model trained & saved as gesture_model.h5")

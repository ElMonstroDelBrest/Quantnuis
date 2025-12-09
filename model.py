import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # pyright: ignore
from tensorflow.keras.models import Sequential  # pyright: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional  # pyright: ignore
from tensorflow.keras.optimizers import Adam  # pyright: ignore

features = []
label = []

df = pd.read_csv('annotation.csv')

for i in range(len(df)):
    audio_path = r'slices/' + str(df['nfile'].values[i])
    original_audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=original_audio, sr=sample_rate).T, axis=0)
    features.append(mels)
    label.append(df['label'].values[i])

temp = np.array([features, label], dtype='object')
data = temp.transpose()

X_ = data[:, 0]
Y = data[:, 1]

X = np.empty([np.shape(X_)[0], len(features[0])])
for i in range(np.shape(X_)[0]):
    X[i] = (X_[i])

Y = to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=123,  test_size=0.2)

model = Sequential(
    [
        Dense(256, input_dim=len(features[0]), activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(2, activation='relu'),
        Dense(len(Y[0]), activation='softmax')
    ]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

model.save('model.h5')

print("Model saved to model.h5")
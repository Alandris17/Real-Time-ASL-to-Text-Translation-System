import os
import cv2
import numpy as np
import mediapipe as mp
import _pickle as cPickle

from sklearn.preprocessing import LabelEncoder

from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Activation
from keras.layers import Add, LayerNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras_tuner as kt

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

def extract_landmarks(image):
    """
    Given an image, extract hand landmarks using MediaPipe Hands.
    Returns a flattened array of landmarks (x, y, z) if a hand is detected,
    otherwise returns None.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    else:
        return None

def load_dataset(dataset_path):
    """
    Loads images from the dataset folder (expects structure: dataset_path/LabelName/image.jpg)
    Extracts hand landmarks for each image.
    Returns:
      - X: A NumPy array of landmark feature vectors.
      - y: A NumPy array of corresponding labels.
      - label_list: The list of label names.
    """
    counter = 0
    X = []
    y = []
    label_list = []
    for label in os.listdir(dataset_path):
        label_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(label_dir):
            print(label_dir, "is not a directory")
            continue
        label_list.append(label)
        print("Loading label:", label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            counter += 1
            image = cv2.imread(file_path)
            if image is None:
                print("Invalid image:", file_path)
                continue
            landmarks = extract_landmarks(image)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label)
    return np.array(X), np.array(y), label_list

dataset_path = "Data_preprocessed"
cache_dir = "landmark_gestures"

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

cache_path = os.path.join(cache_dir, "landmarks_cache.pkl")

if os.path.exists(cache_path):
    print("Loading cached dataset from:", cache_path)
    with open(cache_path, "rb") as f:
        X, y, label_list = cPickle.load(f)
else:
    print("Processing dataset from:", dataset_path)
    X, y, label_list = load_dataset(dataset_path)
    print("Caching dataset for future runs...")
    with open(cache_path, "wb") as f:
        cPickle.dump((X, y, label_list), f)

print("Dataset loaded:", X.shape, y.shape)
print("Labels found:", label_list)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

def build_model(hp):
    model = Sequential()

    model.add(Input(shape=(X.shape[1],)))

    num_layers = hp.Int('num_layers', min_value = 1, max_value = 3, default = 2)

    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=512, max_value=2048, step=256, default=1024)
        reg = hp.Float(f'L2_regularization_{i}', min_value=1e-5, max_value=1e-3, sampling='log', default=1e-4)
        model.add(Dense(units, kernel_regularizer=regularizers.L2(reg)))
        model.add(BatchNormalization())
        model.add(Activation('gelu'))
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.2)
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(len(label_list), activation='softmax'))

    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)

    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='gesture_tuning'
)

tuner.search_space_summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(X, y_encoded, epochs=50, validation_split=0.2, callbacks=[early_stopping])

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:", best_hp.values)

loss, accuracy = best_model.evaluate(X, y_encoded, verbose=0)
print(f"Best model training loss: {loss:.4f}, accuracy: {accuracy:.4f}")

export_dir = "exported_model"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

keras_path = os.path.join(export_dir, "gesture_model.keras")
best_model.save(keras_path)
print(f"Saved Keras model as {keras_path}")
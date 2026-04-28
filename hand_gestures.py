import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

dataset_path = "Data_preprocessed"
labels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
labels.sort()

model_path = "exported_model/gesture_model.keras"
model = tf.keras.models.load_model(model_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       model_complexity=1,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(image):
    """
    Extract hand landmarks from an image using MediaPipe Hands.
    Returns a flattened NumPy array (x, y, z) if a hand is detected, otherwise returns None.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)
        landmarks = results.multi_hand_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    else:
        return None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame)
    if landmarks is not None:
        input_data = np.expand_dims(landmarks, axis=0).astype(np.float32)
        prediction = model.predict(input_data, verbose = False)

        pred_index = np.argmax(prediction)
        pred_label = labels[pred_index]
        pred_score = prediction[0][pred_index]
        
        cv2.putText(frame, f'{pred_label} ({pred_score:.2f})', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
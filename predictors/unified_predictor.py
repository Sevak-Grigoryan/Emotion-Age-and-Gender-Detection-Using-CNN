import os
import cv2
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Base dir:", BASE_DIR)

GENDER_MODEL_PATH  = os.path.join(BASE_DIR, "gender_detection", "models", "model.keras")
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "emotional_detection", "models", "model_16.h5")
AGE_PROTO_PATH     = os.path.join(BASE_DIR, "age_detection_models", "age_deploy.prototxt")
AGE_MODEL_PATH     = os.path.join(BASE_DIR, "age_detection_models", "age_net.caffemodel")
LOG_DIR            = os.path.join(BASE_DIR, "logs")

print("Gender model:", GENDER_MODEL_PATH)
print("Logs dir:", LOG_DIR)

os.makedirs(LOG_DIR, exist_ok=True)
CSV_PATH = os.path.join(LOG_DIR, "unified_log.csv")

GENDER_LABELS  = ['Female', 'Male']
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-25)', '(25-32)', '(32-38)', '(38-43)', '(43-48)', '(48-53)', '(53-60)', '(60-100)']


print("Loading models...")
gender_model  = tf.keras.models.load_model(GENDER_MODEL_PATH)
emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO_PATH, AGE_MODEL_PATH)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def preprocess_gray(face_bgr, size=(48,48)):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size) / 255.0
    gray = np.expand_dims(gray, axis=-1)
    return np.expand_dims(gray, axis=0)

def detect_all():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Age • Gender • Emotion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    logs = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            g_input = preprocess_gray(face, (48,48))
            g_pred = gender_model.predict(g_input, verbose=0)
            gender = GENDER_LABELS[np.argmax(g_pred)]


            e_input = preprocess_gray(face, (48,48))
            e_pred = emotion_model.predict(e_input, verbose=0)
            emotion = EMOTION_LABELS[np.argmax(e_pred)]


            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         (78.4263377603, 87.7689143744, 114.895847746),
                                         swapRB=False)


            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LABELS[np.argmax(age_preds)]


            label = f"{age}, {gender}, {emotion}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)


            logs.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "age": age,
                "gender": gender,
                "emotion": emotion
            })

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv2.destroyAllWindows()

    if logs:
        df = pd.DataFrame(logs)
        df.to_csv(CSV_PATH, index=False)
        print(f"Log saved to {CSV_PATH}")

if __name__ == "__main__":
    detect_all()

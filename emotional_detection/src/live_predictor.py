import sys
import cv2
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

HERE = Path(__file__).resolve()           
EMO_ROOT = HERE.parent.parent               

MODEL_PATH = EMO_ROOT / "models" / "model_16.h5"
LOG_DIR    = EMO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH   = LOG_DIR / "emotion_log.csv"


if not MODEL_PATH.is_file():
    sys.exit(f"[ERROR] Emotion model not found: {MODEL_PATH}\n"
             f"Expected at emotional_detection/models/model_16.h5")

print("Base directory:", HERE.parent)  
print("Project directory:", EMO_ROOT)   

EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


print("Loading emotion model...")
emotion_model = tf.keras.models.load_model(str(MODEL_PATH))

def preprocess_face(bgr, size=(48,48)):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    return np.expand_dims(g[..., None], 0)  

def draw_label(img, text, x, y, scale=0.6, thick=2, pad=4):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    if y - h - 2*pad < 0: 
        y = h + 2*pad
    if x + w + 2*pad > img.shape[1]:  
        x = max(0, img.shape[1] - (w + 2*pad))
    cv2.rectangle(img, (x, y - h - 2*pad), (x + w + 2*pad, y + pad), (0, 0, 0), -1)
    cv2.putText(img, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

def predict_emotion_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    window = "Emotion Detection"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    logs = []

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            x_in = preprocess_face(face)
            pred = emotion_model.predict(x_in, verbose=0)[0]
            emotion = EMOTION_LABELS[int(np.argmax(pred))]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
            draw_label(frame, f"Emotion: {emotion}", x, y - 8)

            logs.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "emotion": emotion
            })

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    if logs:
        pd.DataFrame(logs).to_csv(CSV_PATH, index=False)
        print(f"Log saved to: {CSV_PATH}")

if __name__ == "__main__":
    predict_emotion_webcam()

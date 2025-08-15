import os
import cv2
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

# Load Keras gender model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         
PROJECT_DIR = os.path.dirname(BASE_DIR)                        
MODELS_DIR = os.path.join(PROJECT_DIR,  "models")

print(f"Models directory: {MODELS_DIR}")

gender_model_path = os.path.join(MODELS_DIR, "model.keras")
gender_model = tf.keras.models.load_model(gender_model_path)

# Gender labels
gender_labels = ['Female','Male']

# Log for detected genders
gender_log = []

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_gender():
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open webcam.")
        return

    window_name = "Gender Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48)) / 255.0
            face_resized = np.reshape(face_resized, (1, 48, 48, 1))  # Assuming model expects 48x48x1

            # Predict gender
            gender_pred = gender_model.predict(face_resized, verbose=0)
            gender = gender_labels[np.argmax(gender_pred)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 1)
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Log
            gender_log.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gender": gender
            })

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    video.release()
    cv2.destroyAllWindows()

    # Save log
    if gender_log:
        df = pd.DataFrame(gender_log)
        csv_path = r"C:\Users\Sevak\Desktop\CNN_project\gender_detection\logs\gender_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"Log saved to: {csv_path}")


if __name__ == "__main__":
    detect_gender()

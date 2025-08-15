import cv2
import numpy as np
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH   = os.path.join(BASE_DIR, "test", "test.jpg")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_detection", "models", "model.keras")
EMOTION_MODEL= os.path.join(BASE_DIR, "emotional_detection", "models", "model_16.h5")
AGE_PROTO    = os.path.join(BASE_DIR, "age_detection_models", "age_deploy.prototxt")
AGE_MODEL    = os.path.join(BASE_DIR, "age_detection_models", "age_net.caffemodel")

GENDER_L  = ['Female', 'Male']
EMOTION_L = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
AGE_L     = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

gender_model  = tf.keras.models.load_model(GENDER_MODEL)
emotion_model = tf.keras.models.load_model(EMOTION_MODEL)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
face_det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def prep_gray(bgr, size=(48,48)):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype("float32")/255.0
    return np.expand_dims(g[...,None], 0)  # (1,H,W,1)

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    faces = face_det.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                      scaleFactor=1.2, minNeighbors=4, minSize=(40,40))

    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]

        g = GENDER_L[int(np.argmax(gender_model.predict(prep_gray(face), verbose=0)))]
        e = EMOTION_L[int(np.argmax(emotion_model.predict(prep_gray(face), verbose=0)))]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), (78.4263,87.7689,114.8958), swapRB=False, crop=False)
        age_net.setInput(blob)
        a = AGE_L[int(np.argmax(age_net.forward()[0]))]

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, f"{a}, {g}, {e}", (x, max(15, y-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    out_path = os.path.splitext(IMAGE_PATH)[0] + "_annotated.jpg"
    cv2.imwrite(out_path, img)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()

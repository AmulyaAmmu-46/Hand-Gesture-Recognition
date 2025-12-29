import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("rps_model.h5")
labels = ["rock", "paper", "scissors"]

cap = cv2.VideoCapture(0)

def preprocess(img):
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    return img.reshape(1,128,128,3)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:350, 100:350]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    prediction = model.predict(preprocess(roi))
    gesture = labels[np.argmax(prediction)]

    cv2.rectangle(frame,(100,100),(350,350),(0,255,0),2)
    cv2.putText(frame,gesture,(100,90),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("RPS ML", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

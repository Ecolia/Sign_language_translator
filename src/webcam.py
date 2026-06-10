import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("models/sign_language_model.h5")

CLASS_NAMES = [
    'A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U',
    'V','W','X','Y','Z',
    'del','nothing','space'
]

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    roi = cv2.resize(frame, (64, 64))

    img = roi.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    index = np.argmax(prediction)

    label = CLASS_NAMES[index]

    cv2.putText(
        frame,
        label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
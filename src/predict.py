import cv2
import numpy as np
import sys

from tensorflow.keras.models import load_model

model = load_model("models/sign_language_model.h5")

CLASS_NAMES = [
    'A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U',
    'V','W','X','Y','Z',
    'del','nothing','space'
]

image_path = sys.argv[1]

img = cv2.imread(image_path)

img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

index = np.argmax(prediction)

print("Prediction:", CLASS_NAMES[index])
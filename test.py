import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename


model = tf.keras.models.load_model("recycle_classifier.h5")
print("Model Loaded Successfully!")


with open("class_map.json") as f:
    label_map = json.load(f)


rev_map = {v: k for k, v in label_map.items()}


Tk().withdraw()
file_path = askopenfilename(title="Select Image")

if not file_path:
    print("No file selected.")
    exit()


img = cv2.imread(file_path)
img_resized = cv2.resize(img, (150, 150))
img_array = np.expand_dims(img_resized / 255.0, axis=0)


prob = model.predict(img_array)[0][0]   # probability of class "1"
confidence = max(prob, 1 - prob) * 100

threshold = 75   # minimum confidence required


if confidence < threshold:
    label = "INVALID IMAGE (Not a Plastic Item)"
    print(label)
else:
    if prob > 0.5:
        label = "Recyclable Plastic"
    else:
        label = "Non-Recyclable Plastic"

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")


cv2.putText(img, label, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

cv2.imshow("Prediction", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

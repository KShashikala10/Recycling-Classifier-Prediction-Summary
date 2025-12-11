
# ♻️ Plastic Waste Classification using Deep Learning
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Deep Learning](https://img.shields.io/badge/Model-CNN%20Classifier-yellow)
![Image Classification](https://img.shields.io/badge/Task-Binary%20Image%20Classification-green)
![Classes](https://img.shields.io/badge/Classes-2%20Types-brightgreen)
![Confidence](https://img.shields.io/badge/Confidence%20Threshold-75%25-red)
![Dataset](https://img.shields.io/badge/Dataset-Custom-blueviolet)
![Platform](https://img.shields.io/badge/Platform-Desktop-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Hey, it's **K Shashikala**, your BTech tech friend! 🚀  
Welcome to my **Plastic Waste Classification project**, built using **Deep Learning (CNN)** to classify plastic items as **Recyclable** or **Non-Recyclable**.  
Let’s use AI to help the environment! 🌍💚

---

## ✨ Overview
This deep learning project classifies plastic waste into:

- ♻️ **Recyclable Plastic**  
- ❌ **Non-Recyclable Plastic**

The CNN model provides:
- Class prediction  
- Confidence score  
- Auto-reject of low-confidence images (< 75%)  

Trained Model → `recycle_classifier.h5`  
Class Map → `class_map.json`

---

## 📁 Project Structure
```

Recycling-Classifier-Prediction-Summary/
├── train.py
├── test.py
├── recycle_classifier.h5
├── class_map.json
├── dataset/
│   ├── train/
│   │   ├── recyclable/
│   │   ├── non_recycleable/
│   ├── validation/
│       ├── recyclable/
│       ├── non_recycleable/
└── README.md

````

---

## 🧠 Model Training
Run training:

```bash
python train.py
````

The script:

* Loads dataset
* Performs augmentation
* Builds CNN
* Trains for 20 epochs
* Saves model + class map

---

## 🔍 Image Prediction

Run prediction:

```bash
python test.py
```

Features:

* Tkinter image picker
* Image resized to 150×150
* CNN prediction + confidence score
* OpenCV popup with result
* Rejects predictions < 75% confidence

---

## 🧪 Example Output

```
Prediction: Recyclable Plastic
Confidence: 92.41%
```

---

## 🛠 Requirements

Install dependencies:

```bash
pip install tensorflow numpy opencv-python tk
```

---

## 📘 class_map.json

```json
{
  "non_recycleable": 0,
  "recycleable": 1
}
```

---

## 🚀 Future Improvements

* Add more waste categories
* Deploy using Flask or FastAPI
* Convert to TensorFlow Lite
* Build a mobile app
* Improve accuracy using transfer learning

---

## 🙌 Thank You!
---

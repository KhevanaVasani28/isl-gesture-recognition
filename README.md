# 🤟 Indian Sign Language (ISL) Gesture Recognition & English Translation System

An AI-powered system that recognizes Indian Sign Language gestures (A–Z and 0–9) in real time using computer vision and machine learning, with the goal of translating signs into English text.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

---

## 📌 Overview

This project is part of a larger goal to build a **real-time Indian Sign Language (ISL) to English translation system**. The current module focuses on **static gestures** for:

- **Alphabets:** A–Z
- **Digits:** 0–9

The system uses **MediaPipe** for hand landmark detection and a **Random Forest classifier** for gesture recognition. It supports both **single-hand** and **two-hand** gestures, correctly distinguishing between **left** and **right** hands.

---

## 🎯 Features

- ✅ Real-time gesture recognition via webcam
- ✅ MediaPipe-based hand landmark extraction (21 landmarks per hand, 3D)
- ✅ Left-hand and right-hand detection with proper ordering
- ✅ Data augmentation for improved model generalization
- ✅ Cross-validation for reliable evaluation
- ✅ Hyperparameter tuning (RandomizedSearchCV)
- ✅ Confidence scores for predictions
- ✅ Prediction smoothing for stable output
- ✅ Supports up to 2 hands simultaneously

---

## 🏗️ Project Structure

```
isl-gesture-recognition/
│
├── src/
│   ├── config.py                  # Central configuration (paths, constants)
│   ├── extract_landmarks.py       # Extract landmarks from dataset images
│   ├── landmarks_preprocess.py    # Preprocess + augment + split data
│   ├── train_model.py             # Train Random Forest with tuning & CV
│   ├── realtime_prediction.py     # Real-time webcam prediction
│
├── datasets/                      # Raw gesture images (not uploaded)
├── csv/                           # Extracted landmarks + npy files
├── models/                        # Trained model + label encoder
├── results/                       # Confusion matrix, reports
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

### **Pipeline**

```
Dataset Images (A–Z, 0–9)
        ↓
extract_landmarks.py       → Detect hands via MediaPipe, extract 21 landmarks/hand
        ↓
landmarks_preprocess.py    → Clean, encode, augment, split (train/test)
        ↓
train_model.py             → Cross-validate, tune, train Random Forest
        ↓
realtime_prediction.py     → Real-time webcam prediction
```

### **Landmark Representation**

Each image is converted into a **128-dimensional feature vector**:

| Hand      | Feature Type | Size |
|-----------|--------------|------|
| Left      | type (0) + 21 landmarks × 3 (x,y,z) | 64 |
| Right     | type (1) + 21 landmarks × 3 (x,y,z) | 64 |
| **Total** | | **128** |

- Coordinates are **normalized** relative to the wrist (landmark 0) for translation invariance.
- If a hand is not detected, its slots are filled with **zeros** (with type preserved so the model knows which hand is missing).

---

## 🚀 Getting Started

### **1. Clone the Repository**

```bash
git clone https://github.com/KhevanaVasani28/isl-gesture-recognition.git
cd isl-gesture-recognition
```

### **2. Create a Virtual Environment (Recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Prepare Dataset**

Place your gesture image folders inside:

```
datasets/static_gestures/
├── A/
├── B/
├── ...
├── Z/
├── 0/
├── 1/
├── ...
└── 9/
```

### **5. Run the Pipeline (in order)**

```bash
# Step 1: Extract landmarks from images
python src/extract_landmarks.py

# Step 2: Preprocess, augment, and split data
python src/landmarks_preprocess.py

# Step 3: Train the Random Forest model
python src/train_model.py

# Step 4: (Optional) Analyze the trained model
python src/analyze_model.py

# Step 5: Run real-time prediction
python src/realtime_prediction.py
```

**Controls during real-time prediction:**
- `Q` → Quit
- `R` → Reset prediction smoothing

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Model | RandomForestClassifier |
| Trees | Tuned via RandomizedSearchCV |
| Cross-Validation | Stratified 3-fold |
| Data Augmentation | Rotation, scaling, translation, noise |
| Confidence Threshold | 0.6 (configurable) |
| Classes | 36 (A–Z + 0–9) |

---

## 📊 Results

After training, the following files are generated inside `results/`:

- `rf_classification_report.txt` — Precision, recall, F1-score per class
- `rf_confusion_matrix.png` — Visual confusion matrix

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **OpenCV** — image and video processing
- **MediaPipe** — hand landmark detection
- **NumPy / Pandas** — data manipulation
- **scikit-learn** — Random Forest, cross-validation, tuning
- **Matplotlib / Seaborn** — visualization
- **joblib** — model persistence
- **tqdm** — progress bars

---

## 🔮 Roadmap

- [x] Static gestures (A–Z, 0–9)
- [ ] Dynamic gestures (words)
- [ ] Sentence-level translation
- [ ] Text-to-speech output
- [ ] Mobile app deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👤 Author

**Khevana Vasani**
- GitHub: https://github.com/KhevanaVasani28
- Email: khevanavasani@gmail.com

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) by Google for hand tracking
- [Kaggle](https://www.kaggle.com/) for the ISL dataset
- Open-source community for inspiration and tools

---

## ⭐ If you find this project useful, please give it a star!

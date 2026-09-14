from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATASET_PATH = BASE_DIR / "datasets" / "static_gestures"

# # CSV Folder
CSV_FOLDER = BASE_DIR / "csv"

# # Models Folder
MODEL_FOLDER = BASE_DIR / "models"

# Results Folder
RESULT_FOLDER = BASE_DIR / "results"

# Output CSV
LANDMARK_CSV = CSV_FOLDER / "landmarks_extracted.csv"

# Image Size
IMAGE_SIZE = (224, 224)

# MediaPipe Settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.5

# Preprocessed Files

X_TRAIN = CSV_FOLDER / "X_train.npy"
X_TEST = CSV_FOLDER / "X_test.npy"

Y_TRAIN = CSV_FOLDER / "y_train.npy"
Y_TEST = CSV_FOLDER / "y_test.npy"

# ==========================
# Trained Models and Label Encoder
# ==========================
LABEL_ENCODER = MODEL_FOLDER / "label_encoder.pkl"

RF_MODEL = MODEL_FOLDER / "random_forest_model.pkl"

# ==========================
# Results
# ==========================

CLASSIFICATION_REPORT_RF = RESULT_FOLDER / "rf_classification_report.txt"

RF_CONFUSION_MATRIX = RESULT_FOLDER / "rf_confusion_matrix.png"

#++++++++++++++++++++++++++++++++++++++++++++++++++

# Add these to your existing config.py after the existing configurations

# ===========================
# Data Augmentation Settings
# ===========================
AUGMENTATION_SETTINGS = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': False,  # For hand gestures, flip might change meaning
    'brightness_range': [0.8, 1.2],
}

# ==========================
# Cross-validation Settings
# ==========================
CV_FOLDS = 3
CV_SCORING = 'accuracy'

# ==========================
# Hyperparameter Tuning
# ==========================
HYPERPARAMETER_GRID = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

# ==========================
# Confidence Threshold
# ==========================
CONFIDENCE_THRESHOLD = 0.6

# ==========================
# Augmented Data Paths
# ==========================
AUGMENTED_CSV = CSV_FOLDER / "landmarks_augmented.csv"
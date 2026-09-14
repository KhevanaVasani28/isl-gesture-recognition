# %% ==========================================================
# STEP 1 : Import Libraries
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from sklearn.utils import resample
import random
from tqdm import tqdm
from config import *

# %% ==========================================================
# STEP 2 : Load Landmark CSV
# ==========================================================

df = pd.read_csv(LANDMARK_CSV, low_memory=False, dtype={"label": "string"})

if "label" not in df.columns:
    raise KeyError(f"Expected a 'label' column in {LANDMARK_CSV}, but it was missing.")

# print(df.shape)

# df.head()

# %% ==========================================================
# STEP 3 : Missing Values
# ==========================================================

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

print("\nTotal Missing Values :", missing_values.sum())

# Drop rows with any missing values
if missing_values.sum() > 0:
    df.dropna(inplace=True)
    print("Rows with missing values dropped")

# %% ==========================================================
# STEP 4 : Duplicate Rows
# ==========================================================

duplicates = df.duplicated().sum()

print("Duplicate Rows :", duplicates)

if duplicates > 0:

    df.drop_duplicates(inplace=True)

    print("Duplicates Removed")

print(f"Shape after cleaning: {df.shape}")

# %% ==========================================================
# STEP 5 :  Handle Missing/Empty Labels
# ==========================================================

df["label"] = df["label"].astype("string").str.strip()
missing_label_rows = df["label"].isna() | (df["label"] == "")
if missing_label_rows.any():
    print(f"Dropping {missing_label_rows.sum()} rows with missing/empty labels")
    df = df.loc[~missing_label_rows].copy()

# class_distribution = df["label"].value_counts().sort_index()

# print(class_distribution)

# %% ==========================================================
# STEP 7 : Label Encoding
# ==========================================================

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["label"])

# print(df.head())

joblib.dump(encoder,LABEL_ENCODER)

print("Label Encoder Saved")

# %% ==========================================================
# STEP 7 : DATA AUGMENTATION FUNCTION
# ==========================================================

def augment_landmarks(features, label, augmentation_factor=2):
    """
    Augment landmark data with various transformations.
    
    Args:
        features: Array of landmark features
        label: Class label
        augmentation_factor: Number of augmented samples to generate per original sample
    
    Returns:
        List of augmented feature-label pairs
    """
    augmented_data = []
    
    # Reshape features to (2 hands, 64 features per hand including type)
    # Each hand has: 1 (type) + 21*3 (landmarks) = 64 features
    features_2d = features.reshape(2, 64)
    
    for _ in range(augmentation_factor):
        augmented_features = features_2d.copy()
        
        # Augmentation techniques (apply to both hands separately)
        for hand_idx in range(2):
            # Skip if no hand (-1 type)
            if augmented_features[hand_idx, 0] == -1:
                continue
                
            # Get landmark positions (skip type column)
            landmarks = augmented_features[hand_idx, 1:].reshape(21, 3)
            
            # 1. Random rotation (in 3D space)
            if random.random() > 0.5:
                angle = np.random.uniform(-15, 15) * np.pi / 180
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
                landmarks = np.dot(landmarks, rotation_matrix)
            
            # 2. Random scaling (jitter)
            if random.random() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                landmarks = landmarks * scale
            
            # 3. Random translation (jitter)
            if random.random() > 0.5:
                translation = np.random.uniform(-0.05, 0.05, 3)
                landmarks = landmarks + translation
            
            # 4. Random noise
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.02, landmarks.shape)
                landmarks = landmarks + noise
            
            # Flatten landmarks back
            augmented_features[hand_idx, 1:] = landmarks.flatten()
        
        # Flatten features back to 1D
        augmented_data.append([augmented_features.flatten(), label])
    
    return augmented_data

# %% ==========================================================
# STEP 8 : Apply Data Augmentation
# ==========================================================

print("\nApplying Data Augmentation...")

X = df.drop("label", axis=1)
y = df["label"]

augmented_X = []
augmented_y = []

# Convert to numpy for faster processing
X_np = X.to_numpy(dtype=np.float32)
y_np = y.to_numpy(dtype=np.int64)

# Apply augmentation with progress bar
for i in tqdm(range(len(X_np))):
    features = X_np[i]
    label = y_np[i]
    
    # Always keep original sample
    augmented_X.append(features)
    augmented_y.append(label)
    
    # Generate augmented samples
    aug_samples = augment_landmarks(features, label, augmentation_factor=2)
    for aug_features, aug_label in aug_samples:
        augmented_X.append(aug_features)
        augmented_y.append(aug_label)

# Convert to numpy arrays
augmented_X_np = np.array(augmented_X, dtype=np.float32)
augmented_y_np = np.array(augmented_y, dtype=np.int64)

print(f"Original data size: {len(X_np)}")
print(f"Augmented data size: {len(augmented_X_np)}")
print(f"Augmentation factor: {len(augmented_X_np) / len(X_np):.2f}x")

# %% ==========================================================
# STEP 9 : Split Features and Labels
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    augmented_X_np, augmented_y_np,
    test_size=0.2,
    random_state=42,
    stratify=augmented_y_np
)

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# %% ==========================================================
# STEP 10 : Save the NumPy arrays
# ==========================================================

np.save(X_TRAIN, X_train)
np.save(X_TEST, X_test)
np.save(Y_TRAIN, y_train)
np.save(Y_TEST, y_test)

print("\nTraining & Testing Files Saved")

# %% ==========================================================
# STEP 12 : Summary
# ==========================================================

print("="*50)

print("Preprocessing Completed")

print("="*50)

print(f"Original Features: {X.shape[1]}")
print(f"Augmented Features: {augmented_X_np.shape[1]}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Number of Classes: {len(encoder.classes_)}")

# Print class distribution after augmentation
unique, counts = np.unique(augmented_y_np, return_counts=True)
print("\nAugmented Class Distribution:")
for class_idx, count in zip(unique, counts):
    class_name = encoder.inverse_transform([class_idx])[0]
    print(f"  {class_name}: {count}")


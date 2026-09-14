# %% =====================================================
# STEP 1 : Import Libraries
# =====================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
import time

from config import *

# %% =====================================================
# STEP 2 : Load Dataset
# =====================================================

X_train = np.load(X_TRAIN)
X_test = np.load(X_TEST)
y_train = np.load(Y_TRAIN)
y_test = np.load(Y_TEST)

print("="*60)
print("DATA LOADED SUCCESSFULLY")
print("="*60)
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y_train))}")
print("="*60)

# %% =====================================================
# STEP 3 : Cross-Validation Evaluation
# =====================================================

print("\n" + "="*60)
print("CROSS-VALIDATION EVALUATION")
print("="*60)

# Initialize Random Forest for CV evaluation
rf_cv = RandomForestClassifier(
    n_estimators=50,
    max_depth=20,
    random_state=42,
    n_jobs=2
)

# Subsample for CV to keep it fast
CV_SAMPLE = 20000
if len(X_train) > CV_SAMPLE:
    indices = np.random.choice(len(X_train), CV_SAMPLE, replace=False)
    X_cv = X_train[indices]
    y_cv = y_train[indices]
    print(f"Using {CV_SAMPLE} samples for CV (out of {len(X_train)})")
else:
    X_cv = X_train
    y_cv = y_train

# Perform cross-validation
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    rf_cv, X_train, y_train,
    cv=cv,
    scoring=CV_SCORING,
    n_jobs=1
)

print(f"Cross-Validation Scores ({CV_FOLDS} folds):")
for fold, score in enumerate(cv_scores, 1):
    print(f"  Fold {fold}: {score:.4f}")
print(f"\nMean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %% =====================================================
# STEP 4 : HYPERPARAMETER TUNING with GridSearchCV
# =====================================================

print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

TUNE_SAMPLE = 15000
if len(X_train) > TUNE_SAMPLE:
    idx = np.random.choice(len(X_train), TUNE_SAMPLE, replace=False)
    X_tune, y_tune = X_train[idx], y_train[idx]
    print(f"Using {TUNE_SAMPLE} samples for tuning")
else:
    X_tune, y_tune = X_train, y_train

# Define parameter grid
param_distributions = {
    'n_estimators': randint(100, 200),
    'max_depth': [15, 20, 25],
    'min_samples_split': randint(2, 6),
   # 'min_samples_leaf': [1, 2, 4],
}

# print("Searching over parameter grid...")
# print(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

# # Grid search with 3-fold CV
# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
#     param_grid=param_grid,
#     cv=3,
#     scoring='accuracy',
#     n_jobs=-1,
#     verbose=1
# )

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=2),
    param_distributions=param_distributions,
    n_iter=8,               # only 10 combos
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2,
    random_state=42
)

random_search.fit(X_tune, y_tune)

print(f"Best Parameters: {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")

# start_time = time.time()
# grid_search.fit(X_tune, y_tune)
# tuning_time = time.time() - start_time

# print(f"\nHyperparameter tuning completed in {tuning_time:.2f} seconds")
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best CV Score: {grid_search.best_score_:.4f}")

# %% =====================================================
# STEP 5 : Train Final Model with Best Parameters
# =====================================================

print("\n" + "="*60)
print("TRAINING FINAL MODEL")
print("="*60)

# Create model with best parameters
best_rf = RandomForestClassifier(
    **random_search.best_params_,
    random_state=42,
    n_jobs=2
)

# Train on full training data
print("Training final model...")
best_rf.fit(X_train, y_train)


# Make predictions
y_pred = best_rf.predict(X_test)

# Get prediction probabilities for confidence scores
y_pred_proba = best_rf.predict_proba(X_test)

# %% =====================================================
# STEP 6 : Model Evaluation with Metrics
# =====================================================

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# %% =====================================================
# STEP 7 : Confidence Score Analysis
# =====================================================

# Get max confidence scores for each prediction
confidence_scores = np.max(y_pred_proba, axis=1)

# Categorize predictions by confidence
high = confidence_scores >= CONFIDENCE_THRESHOLD
# medium_confidence = (confidence_scores >= 0.4) & (confidence_scores < CONFIDENCE_THRESHOLD)
# low_confidence = confidence_scores < 0.4
print(f"\nHigh-confidence predictions: {high.mean()*100:.1f}%")

# print(f"High Confidence (≥ {CONFIDENCE_THRESHOLD}): {np.sum(high_confidence)} ({np.mean(high_confidence)*100:.1f}%)")
# print(f"Medium Confidence (0.4-{CONFIDENCE_THRESHOLD}): {np.sum(medium_confidence)} ({np.mean(medium_confidence)*100:.1f}%)")
# print(f"Low Confidence (< 0.4): {np.sum(low_confidence)} ({np.mean(low_confidence)*100:.1f}%)")

# # Calculate accuracy at different confidence thresholds
# thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# print("\nAccuracy by Confidence Threshold:")
# for thresh in thresholds:
#     mask = confidence_scores >= thresh
#     if np.sum(mask) > 0:
#         acc_at_thresh = accuracy_score(y_test[mask], y_pred[mask])
#         coverage = np.mean(mask)
#         print(f"  Threshold {thresh:.1f}: Accuracy={acc_at_thresh:.4f}, Coverage={coverage:.1%}")

# %% =====================================================
# STEP 8 : Classification Report
# =====================================================

report = classification_report(y_test, y_pred, zero_division=0)
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(report)

# Save report
with open(CLASSIFICATION_REPORT_RF, "w") as file:
    file.write("MODEL PERFORMANCE REPORT\n")
    file.write("="*60 + "\n\n")
    file.write(f"Best Parameters: {random_search.best_params_}\n")
    file.write(f"Cross-Validation Mean Score: {cv_scores.mean():.4f}\n\n")
    file.write(report)

print(f"\nClassification report saved to: {CLASSIFICATION_REPORT_RF}")

# %% =====================================================
# STEP 9 : Confusion Matrix
# =====================================================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, xticks_rotation=90)
plt.title(f'Confusion Matrix - Random Forest\nAccuracy: {accuracy:.4f}')
plt.tight_layout()
plt.savefig(RF_CONFUSION_MATRIX, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {RF_CONFUSION_MATRIX}")
plt.show()

# %% =====================================================
# STEP 11 : Save Final Model
# =====================================================
joblib.dump(best_rf, RF_MODEL)
print(f"\nModel saved to: {RF_MODEL}")

# Also save the confidence threshold with the model
model_info = {
    'model': best_rf,
    'best_params': random_search.best_params_,
    'cv_score': cv_scores.mean(),
    'test_accuracy': accuracy,
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'n_features': X_train.shape[1],
    'n_classes': len(np.unique(y_train))
}

joblib.dump(model_info, MODEL_FOLDER / "model_info.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Best parameters: {random_search.best_params_}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Cross-Validation Score: {cv_scores.mean():.4f}")
print("="*60)
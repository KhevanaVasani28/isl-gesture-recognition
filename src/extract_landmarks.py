# %% ==========================================================
# STEP 1 : Import Required Libraries
# ==========================================================

import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

from tqdm import tqdm
from pathlib import Path

from config import *

# %% ==========================================================
# STEP 2 : Initialize MediaPipe Hands
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=True,

    max_num_hands=MAX_NUM_HANDS,

    min_detection_confidence=MIN_DETECTION_CONFIDENCE

)

print("MediaPipe initialized successfully.")

# %% ==========================================================
# STEP 3 : Read All Gesture Classes
# ==========================================================

gesture_classes = sorted([

    folder

    for folder in os.listdir(DATASET_PATH)

    if os.path.isdir(DATASET_PATH / folder)

])

print("Total Classes :", len(gesture_classes))

print(gesture_classes)

# %% ==========================================================
# STEP 4 : Create CSV Header
# ==========================================================

header = []

for hand in range(2):
    header.append(f"hand{hand+1}_type")  # 0 for left, 1 for right, -1 for no hand

    for landmark in range(21):

        header.append(f"hand{hand+1}_x{landmark+1}")

        header.append(f"hand{hand+1}_y{landmark+1}")

        header.append(f"hand{hand+1}_z{landmark+1}")

header.append("label")

print("Total Columns :", len(header))

# %% ==========================================================
# STEP 5 : CORRECTED - Function to Process Hands with Proper Ordering
# ==========================================================

def process_hands_correctly(multi_hand_landmarks, multi_handedness):
    """
    Process hands with proper left/right ordering.
    Returns: (left_hand_landmarks, right_hand_landmarks)
    """
    left_hand = None
    right_hand = None
    
    if multi_hand_landmarks is None:
        return left_hand, right_hand
    
    # Identify which hand is which
    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        handedness = multi_handedness[idx].classification[0].label
        if handedness == 'Left':
            left_hand = hand_landmarks
        else:  # 'Right'
            right_hand = hand_landmarks
    
    return left_hand, right_hand

def extract_hand_features(hand_landmarks, hand_type):
    """
    Extract features for a single hand.
    hand_type: 0 for left, 1 for right
    Returns: list of features [hand_type, x1, y1, z1, x2, y2, z2, ...]
    """
    if hand_landmarks is None:
        # Return zeros for all landmarks, but keep hand_type to indicate which hand it is
        features = [hand_type]  # 0 for left, 1 for right
        features.extend([0.0] * 63)  # 21 landmarks * 3 coordinates
        return features
    
    features = [hand_type]  # 0 for left, 1 for right
    
    # Wrist is landmark 0
    wrist = hand_landmarks.landmark[0]
    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z
    
    # Normalize all landmarks relative to wrist
    for landmark in hand_landmarks.landmark:
        features.extend([
            landmark.x - wrist_x,
            landmark.y - wrist_y,
            landmark.z - wrist_z
        ])
    
    return features

# %% ==========================================================
# STEP 6 : Start Processing Dataset
# ==========================================================
dataset = []

processed = 0

skipped = 0

failed = 0

valid_extensions = (".jpg")

for gesture in tqdm(gesture_classes):

    # Current gesture folder
    folder = DATASET_PATH / gesture

    # Read every image inside the folder
    for image_name in os.listdir(folder):

        # Ignore unwanted files
        if not image_name.lower().endswith(valid_extensions):
            continue

        image_path = folder / image_name

        try:

            # ---------------------------------------------------
            # Read Image
            # ---------------------------------------------------

            image = cv2.imread(str(image_path))

            if image is None:

                failed += 1

                continue

            # ---------------------------------------------------
            # Resize Image
            # ---------------------------------------------------

            image = cv2.resize(image, IMAGE_SIZE)

            # ---------------------------------------------------
            # Convert BGR → RGB
            # ---------------------------------------------------

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------
            # Detect Hands
            # ---------------------------------------------------

            results = hands.process(rgb)

            # ---------------------------------------------------
            # Skip if no hand is detected
            # ---------------------------------------------------

            if results.multi_hand_landmarks is None:

                skipped += 1

                continue


# ---------------------------------------------------
            # CORRECTED: Process with proper left/right handling
            # ---------------------------------------------------
            left_hand, right_hand = process_hands_correctly(
                results.multi_hand_landmarks,
                results.multi_handedness
            )

            # Extract features for both hands
            # Left hand always goes to hand1 position (index 0)
            left_features = extract_hand_features(left_hand, hand_type=0)  # 0 = left
            right_features = extract_hand_features(right_hand, hand_type=1)  # 1 = right

            # Combine: left hand first, right hand second
            features = left_features + right_features

            # Add gesture label
            features.append(gesture)

            # Save one row
            dataset.append(features)
            processed += 1

        except Exception as e:
            failed += 1
            print("Error :", image_path)
            print(e)

            
# %% ==========================================================
# STEP 7 : Processing Statistics
# ==========================================================

print("="*40)

print("Processing Completed")

print("="*40)

print("Processed :", processed)

print("Skipped :", skipped)

print("Failed :", failed)

print("Rows Collected :", len(dataset))

# %% ==========================================================
# STEP 8 : Create DataFrame and Save CSV
# ==========================================================

df = pd.DataFrame(
    dataset,
    columns=header
)
df.head()

df.to_csv(
    LANDMARK_CSV,
    index=False
)

print("CSV Saved Successfully!")

print(LANDMARK_CSV)
# ==================
# Dataset Shape
# ================
print("Dataset Shape: ",df.shape)

# STEP 9 : Verify Hand Type Distribution
# ==========================================================

# Check hand type columns
hand1_types = df['hand1_type'].value_counts()
hand2_types = df['hand2_type'].value_counts()

print("\nHand Type Distribution:")
print(f"Hand 1 (should be left): {hand1_types.to_dict()}")
print(f"Hand 2 (should be right): {hand2_types.to_dict()}")

# %% ==========================================================
# STEP 11 : Label Distribution
# ==========================================================

print(df["label"].value_counts().sort_index())

# %% ==========================================================
# STEP 12 : Close MediaPipe
# ==========================================================

hands.close()

# print("Finished Successfully.")



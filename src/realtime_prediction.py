# %% ==========================================================
# STEP 1 : Import Libraries
# ==========================================================

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

from config import *

# %% ==========================================================
# STEP 2 : Load Model and Label Encoder
# ==========================================================
# Load model info
try:
    model_info = joblib.load(MODEL_FOLDER / "model_info.pkl")
    model = model_info['model']
    confidence_threshold = model_info.get('confidence_threshold', CONFIDENCE_THRESHOLD)
except:
    # Fallback: load model directly
    model = joblib.load(RF_MODEL)
    confidence_threshold = CONFIDENCE_THRESHOLD

print("="*60)
print("MODEL LOADED SUCCESSFULLY")
print("="*60)

# Load label encoder
label_encoder = joblib.load(LABEL_ENCODER)
print(f"\nClasses: {len(label_encoder.classes_)}")
print(label_encoder.classes_)

# %% ==========================================================
# STEP 4 : Initialize MediaPipe Hands
# ==========================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(

    static_image_mode=False,

    max_num_hands=MAX_NUM_HANDS,

    min_detection_confidence=MIN_DETECTION_CONFIDENCE,

    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

print("MediaPipe initialized successfully.")

# %% ==========================================================
# STEP 5 : Landmark Extraction Function
# ==========================================================

def process_hands_correctly(multi_hand_landmarks, multi_handedness):
    """Identify which hand is left and which is right."""
    left_hand = None
    right_hand = None
    
    if multi_hand_landmarks is None:
        return left_hand, right_hand
    
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
    features = [hand_type]  # 0 for left, 1 for right
    
    if hand_landmarks is None:
        # Extend with 63 zeros (21 landmarks * 3)
        features.extend([0.0] * 63)
        return features
    
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

def extract_landmark_features(results):
    """
    Extract normalized hand landmarks from MediaPipe results.
    Always returns: [left_hand_features, right_hand_features]
    where each hand has [hand_type] + 63 landmark coordinates
    """
    if results.multi_hand_landmarks is None:
        return None
    
    # Get left and right hands
    left_hand, right_hand = process_hands_correctly(
        results.multi_hand_landmarks,
        results.multi_handedness
    )
    
    # Extract features for both hands
    # Left hand: type=0, Right hand: type=1
    left_features = extract_hand_features(left_hand, hand_type=0)
    right_features = extract_hand_features(right_hand, hand_type=1)
    
    # Combine: left hand first, then right hand
    features = left_features + right_features
    
    # Convert to NumPy array
    features = np.array(features, dtype=np.float32)
    return features

# %% ==========================================================
# STEP 5 : Smoothing for Stable Predictions
# ==========================================================

class PredictionSmoothing:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
    
    def add_prediction(self, prediction, confidence):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
    
    def get_smoothed_prediction(self):
        if len(self.predictions) == 0:
            return None, 0.0
        
        from collections import Counter
        pred_counts = Counter(self.predictions)
        most_common_pred = pred_counts.most_common(1)[0][0]
        
        avg_conf = np.mean([c for p, c in zip(self.predictions, self.confidences) 
                           if p == most_common_pred])
        
        return most_common_pred, avg_conf

# Initialize smoothing
smoothing = PredictionSmoothing(window_size=5)


# %% ==========================================================
# STEP 9 : Real-Time Gesture Prediction
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError("Could not open webcam.")
else :
    print("Webcam opened successfully.")

print("\nPress 'Q' to quit")
print("Press 'R' to reset smoothing")
print("="*60)

# FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

while True:

    # ----------------------------------------------------------
    # Read frame
    # ----------------------------------------------------------

    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break

    # ----------------------------------------------------------
    # Mirror image
    # ----------------------------------------------------------

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # FPS calculation
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()
    # ----------------------------------------------------------
    # Convert BGR → RGB
    # ----------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------------------
    # Detect hands
    # ----------------------------------------------------------

    results = hands.process(rgb_frame)

    # ----------------------------------------------------------
    # Extract landmarks
    # ----------------------------------------------------------

    features = extract_landmark_features(results)

    prediction_text = "No Hand Detected"
    confidence_text = ""
    prediction_confidence = 0.0
    hand_info = ""
    # ----------------------------------------------------------
    # If hand detected
    # ----------------------------------------------------------

    if features is not None:

        # Model expects 2D input:
        # (samples, features)

        features = features.reshape(1, -1)

        # ------------------------------------------------------
        # Predict gesture number
        # ------------------------------------------------------
        # Get prediction and confidence
        pred_proba = model.predict_proba(features)[0]
        prediction = np.argmax(pred_proba)
        confidence = np.max(pred_proba)

# Apply smoothing
        smoothing.add_prediction(prediction, confidence)
        smoothed_pred, smoothed_conf = smoothing.get_smoothed_prediction()

        if smoothed_pred is not None:
            # Use smoothed prediction
            prediction_text = label_encoder.inverse_transform([smoothed_pred])[0]
            prediction_confidence = smoothed_conf
            
            # Confidence indicator
            if prediction_confidence >= confidence_threshold:
                confidence_text = f"High Confidence ({prediction_confidence:.2f})"
                color = (0, 255, 0)  # Green
            elif prediction_confidence >= 0.4:
                confidence_text = f"Medium Confidence ({prediction_confidence:.2f})"
                color = (0, 255, 255)  # Yellow
            else:
                confidence_text = f"Low Confidence ({prediction_confidence:.2f})"
                color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 255)

        # Determine which hands are detected
        left_hand, right_hand = process_hands_correctly(
            results.multi_hand_landmarks,
            results.multi_handedness
        )
        
        if left_hand is not None and right_hand is not None:
            hand_info = "Both Hands"
        elif left_hand is not None:
            hand_info = "Left Hand Only"
        elif right_hand is not None:
            hand_info = "Right Hand Only"

        # Draw landmarks
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine color based on hand type
                handedness = results.multi_handedness[idx].classification[0].label
                if handedness == 'Left':
                    hand_color = (255, 0, 0)  # Blue for left
                else:
                    hand_color = (0, 0, 255)  # Red for right
                
                mp_draw.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=hand_color, thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
                )

    # Display information on frame
    # Gesture prediction
    cv2.putText(
        display_frame,
        f"Gesture: {prediction_text}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color if 'color' in locals() else (255, 255, 255),
        3
    )

    # Hand info
    if hand_info:
        cv2.putText(
            display_frame,
            f"Hands: {hand_info}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

    # Confidence score
    if confidence_text:
        cv2.putText(
            display_frame,
            confidence_text,
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color if 'color' in locals() else (255, 255, 255),
            2
        )

    # FPS
    cv2.putText(
        display_frame,
        f"FPS: {fps}",
        (30, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    # Instructions
    cv2.putText(
        display_frame,
        "Press Q to Quit | R to Reset Smoothing",
        (30, display_frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    # Show frame
    cv2.imshow("Real-Time ISL Recognition", display_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        smoothing = PredictionSmoothing(window_size=5)
        print("Smoothing reset")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\nReal-time prediction stopped.")
print("="*60)
        
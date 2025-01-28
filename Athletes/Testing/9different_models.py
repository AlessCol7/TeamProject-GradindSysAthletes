import numpy as np
import cv2
import joblib
import tensorflow as tf
from ultralytics import YOLO
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler

# Custom MSE Metric
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

tf.keras.utils.get_custom_objects()["mse"] = mse

# Paths
classification_model_path = "Athletes/KeypointDetection/JsonKeypoints/voting_classifier1.pkl"
scaler_path_classification = "Athletes/KeypointDetection/JsonKeypoints/scaler.pkl"
scoring_models = {
    "Discurweper": ("Athletes/Testing/Discurweper.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Estafette": ("Athletes/Testing/Estafette.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Hoogspringen": ("Athletes/Testing/Hoogspringen.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Hordelopen": ("Athletes/Testing/Hordelopen.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Kogelstoten": ("Athletes/Testing/Kogelstoten.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Speerwerpen": ("Athletes/Testing/Speerwerpen.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "sprint_start": ("Athletes/Testing/sprint_start.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "sprint": ("Athletes/Testing/sprint.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
    "Verspringen": ("Athletes/Testing/Verspringen.h5", "Athletes/KeypointDetection/JsonKeypoints/Testing/scoring_scaler7.pkl"),
}
class_labels = [
    "Discurweper", "Estafette", "Hoogspringen", "Hordelopen",
    "Kogelstoten", "Speerwerpen", "sprint_start", "sprint", "Verspringen"
]

# Load models
voting_clf = joblib.load(classification_model_path)
scaler_classification = joblib.load(scaler_path_classification)
pose_model = YOLO("yolov8n-pose.pt")

def fit_and_save_scaler(data, scaler_path):
    """Fit and save a MinMaxScaler."""
    if len(data) == 0:
        raise ValueError("Training data is empty.")
    scaler = MinMaxScaler()
    scaler.fit(data)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")

def extract_keypoints_from_frame(frame, pose_model, expected_size):
    """Extract keypoints and normalize them."""
    results = pose_model(frame)
    if not results or results[0].keypoints is None:
        return None

    keypoints = results[0].keypoints.xy.cpu().numpy().flatten()
    expected_num_keypoints = 17 * 3  # 17 keypoints with x, y
    normalized_keypoints = np.pad(
        keypoints, (0, max(0, expected_num_keypoints - len(keypoints))), mode="constant"
    )[:expected_num_keypoints]
    return normalized_keypoints


def process_results(results_batch, frame_batch, frame_scores, clf, scaler_classification, class_labels):
    """Process YOLO results, classify activities, and score them."""
    for frame, results in zip(frame_batch, results_batch):
        if len(results) == 0 or results[0].keypoints is None:
            print("No keypoints detected in this frame.")
            continue

        largest_bbox = None
        largest_area = 0
        largest_keypoints = None

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
                # Get the first bounding box and keypoints
                bbox = result.boxes.xyxy.cpu().numpy()[0]
                keypoints = result.keypoints.xy.cpu().numpy().flatten()

                # Ensure bbox is valid (has 4 elements)
                if len(bbox) == 4:
                    # Calculate area of bounding box (width * height)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    # Track the largest bounding box by area
                    if area > largest_area:
                        largest_area = area
                        largest_bbox = bbox
                        largest_keypoints = keypoints
                else:
                    print(f"Bounding box with invalid size: {len(bbox)}")
            else:
                print("Skipping frame due to missing boxes or keypoints.")

        # Process if we found a valid largest person
        if largest_keypoints is not None:
            keypoints = largest_keypoints[:34]  # Ensure only 34 features are used (17 keypoints, x and y for each)
            print(f"Selected largest bounding box: {largest_bbox}")
            print(f"Extracted keypoints: {keypoints}")

            # Ensure the keypoints are exactly 51 features (17 keypoints * 3 values)
            expected_num_keypoints = 51  # 17 keypoints with 3 values (x, y, confidence or other)
            normalized_keypoints = np.pad(
                keypoints, (0, max(0, expected_num_keypoints - len(keypoints))), mode="constant"
            )[:expected_num_keypoints]  # Pad or trim to 51

            print(f"Normalized keypoints: {normalized_keypoints}")

            try:
                # Normalize the keypoints using the scaler
                normalized_keypoints = scaler_classification.transform([normalized_keypoints])

                # Classify the exercise
                exercise_class = clf.predict(normalized_keypoints)[0]

                # Calculate class probabilities
                class_probabilities = softmax(clf.predict_proba(normalized_keypoints), axis=1)

                print(f"Detected exercise class: {class_labels[exercise_class]} (Class ID: {exercise_class})")
                print(f"Class probabilities: {class_probabilities}")

                frame_scores.append((exercise_class, class_probabilities))
            except Exception as e:
                print(f"Error during classification or scoring: {e}")
        else:
            print("No valid keypoints found for the largest person in this frame.")

    return frame_scores


def classify_video(video_path, voting_clf, pose_model, expected_size, class_labels):
    """Classify the activity in a video."""
    cap = cv2.VideoCapture(video_path)
    frame_classes = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints_from_frame(frame, pose_model, expected_size)
        if keypoints is None:
            continue
        keypoints_scaled = scaler_classification.transform([keypoints])
        class_id = voting_clf.predict(keypoints_scaled)[0]
        frame_classes.append(class_id)

    cap.release()
    if not frame_classes:
        raise ValueError("No valid frames with keypoints were extracted.")
    
    predicted_class = np.argmax(np.bincount(frame_classes))
    return class_labels[predicted_class]

def score_video(video_path, scoring_model_path, scaler_path, pose_model, expected_size):
    """Score a classified video."""
    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(scoring_model_path)
    cap = cv2.VideoCapture(video_path)
    frame_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints_from_frame(frame, pose_model, expected_size)
        if keypoints is None:
            continue
        keypoints_scaled = scaler.transform([keypoints])
        score = model.predict(keypoints_scaled).flatten()[0]
        frame_scores.append(np.clip(score, 1, 5))

    cap.release()
    return np.mean(frame_scores) if frame_scores else 0

# Main Script
video_path = "Athletes/Testing/SegmentedVideosOriginal/segment_000185.mp4"
expected_size = 51

# Step 1: Classify the video
predicted_class = classify_video(video_path, voting_clf, pose_model, expected_size, class_labels)
print(f"Predicted Class: {predicted_class}")

# Step 2: Score the video
scoring_model_path, scaler_path = scoring_models[predicted_class]
average_score = score_video(video_path, scoring_model_path, scaler_path, pose_model, expected_size)
print(f"Average Score for {predicted_class}: {average_score:.2f}")

import os
import cv2
import joblib
import numpy as np
from collections import Counter
from keras.models import load_model
from scipy.special import softmax
from ultralytics import YOLO
import tensorflow as tf

# Define and register the custom metric if necessary
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

tf.keras.utils.get_custom_objects()["mse"] = mse

# Define the rounding function to the nearest 0.5
def round_to_nearest_half(x):
    return round(x * 2) / 2

# Define the classification and scoring function with batch processing for YOLO
def classify_and_score(segment_path, segment_name, yolo_model, clf, scaler_classification, scoring_model, scaler_scoring, exercise_labels_inv):
    cap = cv2.VideoCapture(segment_path)
    if not cap.isOpened():
        print(f"Cannot open segment {segment_path}")
        return None

    frame_scores = []
    class_counts = Counter()
    batch_size = 16  # Define batch size for YOLO processing
    frame_batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_batch.append(frame)

        # Process the batch when it reaches batch_size
        if len(frame_batch) == batch_size:
            results_batch = yolo_model(frame_batch)
            frame_scores, class_counts = process_results(results_batch, frame_batch, frame_scores, class_counts, clf, scaler_classification, scoring_model, scaler_scoring)

            frame_batch = []  # Clear the batch

    # Process any remaining frames
    if frame_batch:
        results_batch = yolo_model(frame_batch)
        frame_scores, class_counts = process_results(results_batch, frame_batch, frame_scores, class_counts, clf, scaler_classification, scoring_model, scaler_scoring)

    cap.release()

    # Aggregate results
    average_score = np.mean(frame_scores) if frame_scores else 0
    most_common_class = max(class_counts, key=class_counts.get) if class_counts else None

    return {
        "segment_name": segment_name,
        "average_score": average_score,
        "most_common_class": exercise_labels_inv.get(most_common_class, "Unknown"),
        "class_counts": class_counts,
        "individual_scores": frame_scores,
    }

# Helper function to process YOLO results
def process_results(results_batch, frame_batch, frame_scores, class_counts, clf, scaler_classification, scoring_model, scaler_scoring):
    for frame, results in zip(frame_batch, results_batch):
        if len(results) == 0 or results[0].keypoints is None:
            continue

        # Identify the closest person (largest bounding box)
        largest_bbox = None
        largest_area = 0
        largest_keypoints = None
        for result in results:
            if result.boxes and result.keypoints:
                bbox = result.boxes.xyxy.cpu().numpy()[0]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > largest_area:
                    largest_area = area
                    largest_bbox = bbox
                    largest_keypoints = result.keypoints.xy.cpu().numpy().flatten()

        if largest_keypoints is not None:
            keypoints = largest_keypoints

            # Classify exercise
            if keypoints.size == 34:  # Assuming 17 keypoints with x and y coordinates
                normalized_keypoints = scaler_classification.transform([keypoints])
                normalized_keypoints = normalized_keypoints[:, :9]  # Ensure only 9 features are used
                exercise_class = clf.predict(normalized_keypoints)[0]
                exercise_class_proba = clf.predict_proba(normalized_keypoints)
                exercise_class_proba = softmax(exercise_class_proba, axis=1)
                class_counts[exercise_class] += 1

                # Extract scoring features
                width = largest_bbox[2] - largest_bbox[0]
                height = largest_bbox[3] - largest_bbox[1]
                aspect_ratio = width / height
                keypoints_reshaped = keypoints.reshape(-1, 2)
                distances = [
                    np.linalg.norm(keypoints_reshaped[i] - keypoints_reshaped[j])
                    for i in range(len(keypoints_reshaped))
                    for j in range(i + 1, len(keypoints_reshaped))
                ]
                avg_distance = np.mean(distances) if distances else 0
                area = width * height

                features = np.array([width, height, aspect_ratio, area, avg_distance]).reshape(1, -1)
                features_scaled = scaler_scoring.transform(features)

                # Predict score
                raw_score = scoring_model.predict(features_scaled).flatten()[0]
                scaled_score = np.clip(raw_score, 1, 5)  # Adjusted scaling
                rounded_score = round_to_nearest_half(scaled_score)
                frame_scores.append(rounded_score)

    return frame_scores, class_counts

# Load YOLO model
yolo_model = YOLO("yolov8s-pose.pt")

# Load models and scalers
clf = joblib.load("Athletes/KeypointDetection/JsonKeypoints/optimized_rf_model.pkl")
scaler_classification = joblib.load("Athletes/Testing/scaler8.pkl")
exercise_labels_inv = joblib.load("Athletes/KeypointDetection/JsonKeypoints/Testing/exercise_labels_again.pkl")
scoring_model = tf.keras.models.load_model("Athletes/KeypointDetection/JsonKeypoints/scoring_model.h5")
scaler_scoring = joblib.load("Athletes/KeypointDetection/JsonKeypoints/scoring_scaler.pkl")

# Specify the path to the segmented videos folder
segment_folder = "Athletes/Testing/SegmentedVideos1"
segments = [os.path.join(segment_folder, f) for f in os.listdir(segment_folder) if f.endswith('.mp4')]

# Process each video segment
segment_results = {}
for segment in segments:
    segment_name = os.path.basename(segment)
    results = classify_and_score(segment, segment_name, yolo_model, clf, scaler_classification, scoring_model, scaler_scoring, exercise_labels_inv)
    if results:
        segment_results[segment_name] = results

# Calculate and display results
class_scores = {}
for segment_name, result in segment_results.items():
    class_name = result['most_common_class']
    if class_name not in class_scores:
        class_scores[class_name] = {'scores': [], 'count': 0, 'segment_names': []}
    class_scores[class_name]['scores'].append(result['average_score'])
    class_scores[class_name]['count'] += 1
    class_scores[class_name]['segment_names'].append(result['segment_name'])

# Aggregate and print final results
for class_name, data in class_scores.items():
    average_class_score = np.mean(data['scores'])
    rounded_scores = [round_to_nearest_half(score) for score in data['scores']]
    print(f"Class: {class_name}")
    print(f"  Average Score for this class: {average_class_score:.2f}")
    print(f"  Number of Segments: {data['count']}")
    print(f"  Scores from each segment: {data['scores']}")
    print(f"  Rounded scores from each segment: {rounded_scores}")
    print(f"  Segments: {data['segment_names']}\n")

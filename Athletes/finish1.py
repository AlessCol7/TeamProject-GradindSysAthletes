import os
import json
import cv2
import joblib
from ultralytics import YOLO
import numpy as np
import logging

# --------------------- Configuration ---------------------
YOLO_MODEL_PATH = "yolov8n-pose.pt"  # Path to your YOLOv8 pose model
CLASSIFIER_MODEL_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/exercise_classifier.pkl"
SCALER_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/scaler.pkl"
DOWNLOAD_FOLDER = "/path/to/DownloadVideoTest"
SEGMENT_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/SegmentedVideos"
KEYPOINT_JSON_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/annotations.json"
CONFIDENCE_THRESHOLD = 0.2  # Confidence threshold for pose detection

# --------------------- Setup Logging ---------------------
logging.basicConfig(level=logging.INFO)

# --------------------- YOLO and Classifier Models ---------------------
yolo_model = YOLO(YOLO_MODEL_PATH)
classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logging.info("Classifier and scaler loaded successfully.")

# --------------------- Helper Functions ---------------------
def convert_to_native_types(data):
    """Convert numpy types to native Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to a Python list
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)  # Convert numpy int to Python int
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)  # Convert numpy float to Python float
    return data

def recursively_convert(data):
    """Recursively convert all values in nested structures to native Python types."""
    if isinstance(data, dict):
        return {key: recursively_convert(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [recursively_convert(element) for element in data]
    else:
        return convert_to_native_types(data)

# --------------------- Keypoint Detection and Classification ---------------------
def process_segments(segments, yolo_model, classifier_model, scaler):
    annotations = {"segments": {}}

    for segment_path in segments:
        cap = cv2.VideoCapture(segment_path)
        if not cap.isOpened():
            logging.error(f"Cannot open segment {segment_path}")
            continue

        frame_count = 0
        segment_annotations = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Only process every 5th frame
            if frame_count % 5 != 0:
                continue

            # Run YOLO pose detection
            results = yolo_model(frame)
            largest_area = 0
            selected_keypoints = None
            bbox = None

            # Iterate over detected persons (if any)
            for result in results:
                if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
                    keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 2).flatten()
                    detected_bbox = result.boxes.xyxy.cpu().numpy()[0]
                    area = (detected_bbox[2] - detected_bbox[0]) * (detected_bbox[3] - detected_bbox[1])

                    # Select the person with the largest bounding box
                    if area > largest_area:
                        largest_area = area
                        selected_keypoints = keypoints
                        bbox = detected_bbox

            # If keypoints are selected, process them
            if selected_keypoints is not None:
                try:
                    if len(selected_keypoints) >= 34:
                        selected_keypoints = selected_keypoints[:34]
                    selected_keypoints = selected_keypoints.astype(float)

                    # Normalize keypoints using the scaler
                    normalized_keypoints = scaler.transform([selected_keypoints])
                    normalized_keypoints = normalized_keypoints.astype(float)

                    # Predict class probabilities
                    probabilities = classifier_model.predict_proba(normalized_keypoints)[0]
                    predicted_class = classifier_model.classes_[np.argmax(probabilities)]

                    # Prepare annotation
                    annotation = {
                        "bbox": convert_to_native_types(bbox),
                        "keypoints": convert_to_native_types(selected_keypoints),
                        "normalized_keypoints": convert_to_native_types(normalized_keypoints),
                        "probabilities": convert_to_native_types(probabilities),
                        "predicted_class": predicted_class,
                        "bbox_area": convert_to_native_types((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    }
                    segment_annotations.append(annotation)
                except Exception as e:
                    logging.error(f"Error processing keypoints: {e}")

        # Store segment annotations
        annotations["segments"][os.path.basename(segment_path)] = {
            "annotations": segment_annotations
        }
        cap.release()

    # Convert and save annotations to JSON
    with open(KEYPOINT_JSON_PATH, "w") as f:
        json.dump(recursively_convert(annotations), f, indent=4)
    logging.info(f"Annotations saved to {KEYPOINT_JSON_PATH}")

# --------------------- Main Program ---------------------
if __name__ == "__main__":
    # List all segmented video files in the segment folder
    segments = [os.path.join(SEGMENT_FOLDER, f) for f in os.listdir(SEGMENT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    if segments:
        process_segments(segments, yolo_model, classifier_model, scaler)
    else:
        logging.error("No segmented videos found. Exiting.")

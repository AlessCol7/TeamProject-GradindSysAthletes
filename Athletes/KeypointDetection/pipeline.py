import os
import cv2
import json
from ultralytics import YOLO
import joblib  # For loading classification models

# Paths
INPUT_VIDEO_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/YouTubeDownload/DownloadedVideos/Video Atletiek.mp4"  # Single video path
SEGMENTED_VIDEO_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/NewSegementedVideos"
CLASSIFICATION_RESULTS_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/ClassificationFolder"
MODEL_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/exercise_classifier.pkl"  # Classification model
YOLO_MODEL_PATH = "yolov8s-pose.pt"  # YOLO pose model

# Ensure output folders exist
os.makedirs(SEGMENTED_VIDEO_FOLDER, exist_ok=True)
os.makedirs(CLASSIFICATION_RESULTS_FOLDER, exist_ok=True)

# Load models
yolo_model = YOLO(YOLO_MODEL_PATH)
classifier_model = joblib.load(MODEL_PATH)

def segment_video(video_path, output_folder):
    """Segment video using YOLO pose detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    frame_count = 0
    is_recording = False
    video_writer = None
    segments = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = yolo_model(frame)

        # Check if any person is detected
        detection_exists = any(result.keypoints is not None for result in results)

        # Start a new segment if detection begins
        if detection_exists and not is_recording:
            is_recording = True
            segment_name = f"segment_{frame_count:06d}.mp4"
            segment_path = os.path.join(output_folder, segment_name)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(segment_path, fourcc, 30.0, (width, height))
            segments.append(segment_path)
            print(f"Started new segment: {segment_name}")

        # Stop recording if no detection
        if not detection_exists and is_recording:
            is_recording = False
            video_writer.release()
            print(f"Saved segment: {segment_name}")

        # Write frame if recording
        if is_recording and video_writer:
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()

    return segments

def classify_segments(segments, model, output_folder):
    """Classify segmented videos."""
    results = []
    for segment in segments:
        # Extract features from the video (placeholder for feature extraction)
        features = extract_features_from_video(segment)

        # Predict the class
        prediction = model.predict([features])[0]
        result = {"segment": segment, "class": prediction}
        results.append(result)

        print(f"Classified {segment}: {prediction}")

        # Save the result
        result_path = os.path.join(output_folder, os.path.basename(segment) + ".json")
        with open(result_path, 'w') as f:
            json.dump(result, f)

    return results

def extract_features_from_video(video_path):
    """Extract features from video for classification."""
    # Placeholder for feature extraction logic
    # Example: Using keypoints, motion, or frame histograms
    return [0] * 100  # Dummy feature vector

def process_single_video(video_path, output_folder, model):
    """Process a single video."""
    print(f"Processing video: {video_path}")

    # Step 1: Segment video
    segmented_videos = segment_video(video_path, output_folder)

    # Step 2: Classify segments
    classify_segments(segmented_videos, model, CLASSIFICATION_RESULTS_FOLDER)

# Run the pipeline on a single video
process_single_video(INPUT_VIDEO_PATH, SEGMENTED_VIDEO_FOLDER, classifier_model)

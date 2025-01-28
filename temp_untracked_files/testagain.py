import os
import json
import cv2
import joblib
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import numpy as np
import logging

# --------------------- Configuration ---------------------
YOLO_MODEL_PATH = "yolov8s-pose.pt"  # Path to your YOLOv8 pose model
CLASSIFIER_MODEL_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/exercise_classifier.pkl"  # Path to your saved classifier model
DOWNLOAD_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/DownloadVideoTest"
SEGMENT_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/SegmentedVideos"
KEYPOINT_JSON_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/annotations.json"
CONFIDENCE_THRESHOLD = 0.2  # Adjust based on your needs

# --------------------- Setup Logging ---------------------
logging.basicConfig(level=logging.INFO)

# --------------------- YOLO and Classifier Models ---------------------
if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}. Please download or specify the correct path.")
yolo_model = YOLO(YOLO_MODEL_PATH)

# Classifier Model Initialization (scikit-learn)
classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)

# --------------------- Video Download ---------------------
def download_video(video_link, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    options = {
        'format': 'best',
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
    }

    with YoutubeDL(options) as ydl:
        info_dict = ydl.extract_info(video_link, download=True)
        return ydl.prepare_filename(info_dict)

# --------------------- Segmentation ---------------------
def segment_video(video_path, segment_folder):
    if not os.path.exists(segment_folder):
        os.makedirs(segment_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return []

    frame_count = 0
    is_recording = False
    video_writer = None
    segment_annotations = []
    segments = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Run YOLO object detection
        results = yolo_model(frame)

        closest_bbox = None
        max_bbox_area = 0

        for result in results:
            if result.boxes:
                bbox = result.boxes.xyxy.cpu().numpy()[0]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if area > max_bbox_area:
                    max_bbox_area = area
                    closest_bbox = bbox

        if closest_bbox is not None:
            if not is_recording:
                is_recording = True
                output_path = os.path.join(segment_folder, f"segment_{frame_count + 1:06d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                print(f"Started recording: {output_path}")

            x_min, y_min, x_max, y_max = closest_bbox
            segment_annotations.append({
                "id": frame_count,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": max_bbox_area,
            })

        elif is_recording:
            is_recording = False
            video_writer.release()

            segment_filename = os.path.basename(output_path)
            segments.append({
                "filename": segment_filename,
                "annotations": segment_annotations
            })
            segment_annotations = []
            print(f"Stopped recording: {segment_filename}")

        if is_recording:
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()

    return segments

# --------------------- Keypoint Detection, Classification, and JSON Annotations ---------------------
def apply_keypoint_detection_on_segments(segments, segment_folder, yolo_model, classifier_model):
    segment_annotations = {}

    # COCO Keypoints order and Skeleton structure
    COCO_KEYPOINTS = [
        'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
        'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
        'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
        'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
    ]
    SKELETON = [
        [0, 1], [1, 3], [3, 5], [0, 2], [2, 4], [4, 6],
        [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    # Initialize the COCO JSON structure
    coco_annotations = {
        "segments": {},
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": COCO_KEYPOINTS,
                "skeleton": SKELETON
            }
        ]
    }

    for segment in segments:
        segment_filename = segment["filename"]
        segment_path = os.path.join(segment_folder, segment_filename)

        cap = cv2.VideoCapture(segment_path)
        if not cap.isOpened():
            print(f"Error: Unable to open {segment_path}. Skipping this segment.")
            continue  # Skip this segment and move to the next

        frame_count = 0
        segment_annotations[segment_filename] = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            # Run YOLO pose detection
            results = yolo_model(frame)

            closest_bbox = None
            closest_keypoints = None
            max_bbox_area = 0

            for result in results:
                if result.keypoints and result.boxes:
                    keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 3)  # (N, 3) for each keypoint (x, y, confidence)
                    bbox = result.boxes.xyxy.cpu().numpy()[0]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    if area > max_bbox_area:
                        max_bbox_area = area
                        closest_bbox = bbox
                        closest_keypoints = keypoints[:, :2]  # Only keep x, y coordinates (exclude confidence)

            if closest_keypoints is not None:
                keypoints_with_visibility = []

                for x, y in closest_keypoints:
                    keypoints_with_visibility.extend([float(x), float(y), 2])  # Ensure float values

                # Apply classification based on keypoints
                classification_input = np.array(keypoints_with_visibility).reshape(1, -1)  # Flatten keypoints for classifier
                classification_score = classifier_model.predict_proba(classification_input)[0]
                predicted_class = classifier_model.classes_[np.argmax(classification_score)]

                segment_annotations[segment_filename].append({
                    "frame_id": frame_count,
                    "bbox": closest_bbox.tolist(),
                    "keypoints": keypoints_with_visibility,
                    "num_keypoints": len(COCO_KEYPOINTS),
                    "predicted_class": predicted_class,
                    "classification_score": classification_score.tolist()  # List of probabilities for each class
                })

                # Print keypoint and classification data
                print(f"Frame {frame_count} from segment {segment_filename} has keypoints and predicted class '{predicted_class}' with score {classification_score}.")
            else:
                print(f"No keypoints detected in frame {frame_count} of segment {segment_filename}.")

        cap.release()

    # Save annotations to JSON
    with open(KEYPOINT_JSON_PATH, "w") as f:
        json.dump(coco_annotations, f, indent=4)


# --------------------- Main Program ---------------------
if __name__ == "__main__":
    video_input = input("Enter YouTube link or local video file path: ").strip()

    if video_input.startswith("http"):
        video_path = download_video(video_input, DOWNLOAD_FOLDER)
    else:
        video_path = video_input

    if video_path and os.path.exists(video_path):
        segments = segment_video(video_path, SEGMENT_FOLDER)
        apply_keypoint_detection_on_segments(segments, SEGMENT_FOLDER, yolo_model, classifier_model)
    else:
        logging.error("No valid video provided or video file not found. Exiting.")
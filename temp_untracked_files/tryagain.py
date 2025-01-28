import os
import json
import cv2
import joblib  # For loading scikit-learn model
from yt_dlp import YoutubeDL
from ultralytics import YOLO
from bs4 import BeautifulSoup  # For parsing HTML
import requests  # For fetching HTML content
import numpy as np
import jsonpickle

# --------------------- Configuration ---------------------
YOLO_MODEL_PATH = "yolov8s-pose.pt"  # Path to your YOLOv8 pose model
CLASSIFIER_MODEL_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/exercise_classifier.pkl"  # Path to your saved classifier model
DOWNLOAD_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/DownloadVideoTest"
SEGMENT_FOLDER = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/SegmentedVideos"
KEYPOINT_JSON_PATH = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/annotations.json"

# YOLO Model Initialization
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
        try:
            info_dict = ydl.extract_info(video_link, download=True)
            video_path = ydl.prepare_filename(info_dict)
            print(f"Video downloaded: {video_path}")
            return video_path
        except Exception as e:
            print(f"Error downloading video: {e}")

def extract_video_link_from_html(html_file):
    """Extract video link from an HTML file."""
    try:
        with open(html_file, 'r') as file:
            soup = BeautifulSoup(file, 'html.parser')
        
        # Try to find the first YouTube link or other video links
        video_link = None
        for a_tag in soup.find_all('a', href=True):
            if 'youtube.com/watch' in a_tag['href'] or 'youtu.be/' in a_tag['href']:
                video_link = a_tag['href']
                break
        
        if video_link:
            print(f"Extracted video link from HTML: {video_link}")
            return video_link
        else:
            print("No valid video link found in the HTML file.")
            return None
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return None

# --------------------- Segmentation and Classification ---------------------
def convert_types(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, np.int64):
        return int(obj)  # Convert numpy int64 to native Python int
    elif isinstance(obj, np.float64):
        return float(obj)  # Convert numpy float64 to native Python float
    return obj

def segment_detect_classify(video_path, segment_folder, yolo_model, classifier_model):
    if not os.path.exists(segment_folder):
        os.makedirs(segment_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0
    is_recording = False
    video_writer = None
    segment_annotations = []
    segments = {}

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
                keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 2)
                bbox = result.boxes.xyxy.cpu().numpy()[0]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if area > max_bbox_area:
                    max_bbox_area = area
                    closest_bbox = bbox
                    closest_keypoints = keypoints

        if closest_bbox is not None:
            # Preprocess keypoints for classifier model (though we won't use classification score)
            keypoints_input = closest_keypoints.flatten().reshape(1, -1)  # Scikit-learn expects 2D array

            # Ensure we only pass exactly 34 features (17 keypoints * 2)
            if keypoints_input.size != 34:
                print(f"Warning: Keypoints size is {keypoints_input.size}. Expected 34 features.")
                continue  # Skip this frame if the input size is incorrect

            # Start recording if not already
            if not is_recording:
                is_recording = True
                output_path = os.path.join(segment_folder, f"segment_{frame_count:06d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                print(f"Started recording: {output_path}")

            # Append annotation to the current segment
            x_min, y_min, x_max, y_max = closest_bbox
            segment_annotations.append({
                "id": frame_count,  # Unique frame identifier for each annotation
                "image_id": frame_count,  # This can be adjusted based on your system's ID structure
                "category_id": 1,  # Category ID for the exercise (adjust as needed)
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": area,
                "keypoints": [convert_types(k) for k in closest_keypoints.flatten().tolist()],
                "num_keypoints": len(closest_keypoints)
            })

        elif is_recording:
            # Stop recording when person is no longer detected
            is_recording = False
            video_writer.release()

            segment_filename = f"segment_{frame_count:06d}.mp4"
            segments[segment_filename] = {
                "annotations": segment_annotations
            }
            segment_annotations = []  # Reset annotations for the next segment
            print(f"Stopped recording for segment {frame_count}")

        if is_recording:
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()

    # Save segments as a JSON object
    with open(KEYPOINT_JSON_PATH, "w") as f:
        json.dump({"segments": segments}, f, indent=4)



# --------------------- Main Program ---------------------
if __name__ == "__main__":
    print("Welcome to the Video Segmentation and Classification Tool!")
    print("You can provide a YouTube video link or the path to a local video file.")
    print("For YouTube links, the video will be downloaded automatically.")
    
    video_input = input("Enter YouTube link or local video file path: ").strip()

    if video_input.startswith("http"):
        print("Downloading video from the provided YouTube link...")
        video_path = download_video(video_input, DOWNLOAD_FOLDER)
    else:
        print("Using the provided local video file...")
        video_path = video_input

    if video_path and os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        segment_detect_classify(video_path, SEGMENT_FOLDER, yolo_model, classifier_model)
    else:
        print("No valid video provided or video file not found. Exiting.")

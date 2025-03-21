import cv2
import os
import json
from ultralytics import YOLO

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8s-pose.pt")  # Replace with your model path if different

# Define the path to the video folder and output folder
video_folder_path = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/exercises/sprint"
output_json_path = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore/sprint.json"

# Define the COCO keypoints order
COCO_KEYPOINTS = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]

# Skeleton structure connecting keypoints
SKELETON = [
    [0, 1], [1, 3], [3, 5], [0, 2], [2, 4], [4, 6],
    [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
]

# JSON structure
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

# List all video files in the folder and sort them numerically
video_files = sorted([f for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))], key=lambda x: int(''.join(filter(str.isdigit, x))))

for video_file in video_files:
    video_path = os.path.join(video_folder_path, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        continue

    frame_count = 0
    image_id = 0
    annotation_id = 0
    segment_score = 0.95  # Placeholder score for the segment

    segment_annotations = []

    # Process video frames
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        image_id += 1

        # Run YOLO pose detection
        results = model(frame)

        # Initialize variables for the closest person
        closest_bbox = None
        closest_keypoints = None
        max_bbox_area = 0

        for result in results:
            if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
                keypoints = result.keypoints.xy.cpu().numpy().reshape(-1, 2)
                bbox = result.boxes.xyxy.cpu().numpy()[0]
                x_min, y_min, x_max, y_max = bbox
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                area = bbox_width * bbox_height

                # Update the closest person if this bounding box has a larger area
                if area > max_bbox_area:
                    max_bbox_area = area
                    closest_bbox = bbox
                    closest_keypoints = keypoints

        # Save only the closest person annotation every 5 frames
        if frame_count % 5 == 0 and closest_bbox is not None:
            x_min, y_min, x_max, y_max = closest_bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            area = bbox_width * bbox_height
            keypoints_with_visibility = []

            for x, y in closest_keypoints:
                keypoints_with_visibility.extend([float(x), float(y), 2])  # Ensure float values

            # Add the closest person's annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x_min), float(y_min), float(bbox_width), float(bbox_height)],  # Ensure bbox values are floats
                "area": float(area),  # Ensure area is float
                "keypoints": keypoints_with_visibility,
                "num_keypoints": len(COCO_KEYPOINTS)
            }
            annotation_id += 1
            segment_annotations.append(annotation)

            # Save the current annotations to the segment
            coco_annotations["segments"][video_file] = {
                "score": segment_score,
                "annotations": segment_annotations
            }

            # Write the updated JSON file
            with open(output_json_path, "w") as f:
                json.dump(coco_annotations, f, indent=4)

            print(f"Processed frame {frame_count} from {video_file}, JSON updated.")

    cap.release()

print(f"COCO annotations saved to {output_json_path}")

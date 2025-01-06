import cv2
import os
from ultralytics import YOLO

# Initialize the YOLOv8n-pose model
model = YOLO("yolov8s-pose.pt")  # Replace with your model path if different

# Define the path to the video file and output folder
video_path = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/YouTubeDownload/DownloadedVideos/Filmpje sport.mp4"
output_video_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/cutvideos"

# Ensure output folder exists
os.makedirs(output_video_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit()

frame_count = 0
image_id = 0
annotation_id = 0

# Initialize video writer and detection tracking variables
is_recording = False
current_output_video = None
video_writer = None

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

    # If detection exists and not recording, start a new recording
    if closest_bbox is not None and not is_recording:
        is_recording = True
        output_video_name = f"segment_{frame_count:06d}.mp4"
        output_video_path = os.path.join(output_video_folder, output_video_name)

        # Initialize the video writer with the current frame's size
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec for .mp4
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

        # Start recording the segment
        print(f"Started new video segment: {output_video_name}")

    # If no detection and recording, stop recording
    if closest_bbox is None and is_recording:
        is_recording = False
        video_writer.release()
        print(f"Video segment saved as {output_video_name}")

    # If recording, write the current frame to the video file
    if is_recording:
        video_writer.write(frame)

    # Once recording stops, save the segment
    if not is_recording and video_writer:
        video_writer.release()

cap.release()

print(f"Video segmentation completed. All segments are saved in {output_video_folder}")

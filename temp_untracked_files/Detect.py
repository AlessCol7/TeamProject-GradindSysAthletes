import cv2
from ultralytics import YOLO
import numpy as np

# Initialize the YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Load video file
video_path = "YouTubeDownload/DownloadedVideos/atletiek.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file at {video_path}")
    exit()

# Parameters for motion detection and frame skipping
previous_frame = None
motion_threshold = 20
frame_skip = 5  # Process every 5th frame
frame_count = 0

# Process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames for processing

    # Convert frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute frame difference for motion detection
    if previous_frame is not None:
        frame_diff = cv2.absdiff(previous_frame, gray_frame)
        _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

        # Expand detection area to include small/far objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea, default=None)

        if largest_contour is not None:
            # Get bounding box for the moving object
            x, y, w, h = cv2.boundingRect(largest_contour)
            x, y, w, h = max(0, x - 10), max(0, y - 10), w + 20, h + 20  # Expand ROI

            # Check if the bounding box is large enough
            if w * h > 100:  # Threshold for minimum area
                roi = frame[y:y+h, x:x+w]  # Crop the region of interest
            else:
                roi = frame  # Use the full frame if the ROI is too small
        else:
            roi = frame  # Use the full frame if no motion is detected
    else:
        roi = frame  # Use the full frame for the first frame

    # Perform keypoint detection on the selected region
    results = model(roi, save=False)

    # Annotate frame with keypoints
    annotated_frame = results[0].plot()

    # Overlay the annotated region back onto the original frame
    if roi.shape[:2] != frame.shape[:2]:  # Check if ROI is a subregion
        frame[y:y+h, x:x+w] = annotated_frame
    else:
        frame = annotated_frame

    # Update previous frame for motion detection
    previous_frame = gray_frame.copy()

    # Display the processed frame
    cv2.imshow("Athlete Keypoint Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

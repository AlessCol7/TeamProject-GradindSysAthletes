import cv2
from ultralytics import YOLO
import numpy as np
from sort import Sort
import time

# Initialize the YOLOv8 pose model
model = YOLO("yolov8s-pose.pt")

# Load video file
video_path = "/Users/alessiacolumban/Desktop/Athletes/YouTubeDownload/DownloadedVideos/Filmpje sport.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file at {video_path}")
    exit()
else:
    print(f"Successfully opened video file at {video_path}")

# Parameters
frame_skip = 5  # Process every 5th frame
frame_count = 0
roi_margin = 10  # Margin around the athlete's bounding box
min_bbox_size = 500  # Minimum bounding box area (to filter out irrelevant objects)
max_search_frames = 20  # Maximum number of frames to search for the athlete after they disappear
confidence_threshold = 0.5  # Higher confidence threshold for detection
smoothing_factor = 0.5  # Smoothing factor for keypoint detection

# Tracking variables
tracker = Sort()
athlete_track = None
detection_counter = 0  # Counter to track how many frames the athlete has been consistently detected
lost_frames = 0  # To count frames where athlete detection is lost
athlete_missing_frames = 0  # Count frames where the athlete is missing (for 20 frame wait)
previous_athlete_center = None  # Store previous position of the athlete

# Optical Flow Parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Helper function to prioritize athlete detection (only one person is detected per frame)
def prioritize_athlete(detections, tracked_objects, prev_athlete_center):
    best_track = None
    min_distance = float('inf')

    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # If there's a previously tracked athlete, check the proximity to their previous position
        if prev_athlete_center:
            distance = np.sqrt((bbox_center[0] - prev_athlete_center[0]) ** 2 +
                               (bbox_center[1] - prev_athlete_center[1]) ** 2)
        else:
            # If no previous athlete, calculate distance to center of the frame
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Use the center of the frame if no previous position
            distance = np.sqrt((bbox_center[0] - frame_center[0]) ** 2 +
                               (bbox_center[1] - frame_center[1]) ** 2)

        # Select the closest person
        if distance < min_distance:
            min_distance = distance
            best_track = (x1, y1, x2, y2)

    return best_track

# Define how many frames the athlete must be closest to "stick" to them
min_frames_to_stick = 30  # Modify this based on how many frames you want to stick to the athlete

# Process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Failed to read a frame.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames for processing

    # Run YOLO detection
    results = model(frame, conf=confidence_threshold, save=False)
    detections = []

    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].tolist()
            conf = bbox.conf[0].item()
            if (x2 - x1) * (y2 - y1) >= min_bbox_size:
                detections.append([x1, y1, x2, y2, conf])

    # Convert detections to numpy array
    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Update tracker with detections
    tracked_objects = tracker.update(detections)

    if len(tracked_objects) > 0:
        # Get the closest person
        prev_athlete_center = ((athlete_track[0] + athlete_track[2]) // 2, (athlete_track[1] + athlete_track[3]) // 2) if athlete_track is not None else None
        athlete_track = prioritize_athlete(detections, tracked_objects, prev_athlete_center)
        detection_counter += 1  # Increment counter when athlete is detected

        # If we have detected the athlete consistently for enough frames, stick to this person
        if detection_counter >= min_frames_to_stick:
            print("Athlete detected consistently, sticking to this person.")
            lost_frames = 0
        else:
            lost_frames += 1  # If athlete is not detected consistently, stop tracking them
    else:
        athlete_track = None
        lost_frames += 1

    if athlete_track is None and lost_frames >= max_search_frames:
        athlete_track = None
        detection_counter = 0  # Reset detection counter if athlete is missing for too long

    # Process keypoint detection for the athlete
    if athlete_track is not None:
        x1, y1, x2, y2 = map(int, athlete_track[:4])

        # Draw bounding box around athlete
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Athlete", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract ROI around athlete for keypoint detection
        athlete_roi = frame[max(0, y1 - roi_margin):min(frame.shape[0], y2 + roi_margin),
                            max(0, x1 - roi_margin):min(frame.shape[1], x2 + roi_margin)]

        # Run keypoint detection on the athlete's ROI
        # Initialize prev_keypoints
        prev_keypoints = None
        smoothing_factor = 0.5  # Adjust the smoothing factor as needed

        # Inside your main loop
        keypoint_results = model(athlete_roi, save=False)
        if len(keypoint_results[0].keypoints) > 0:
            keypoints = keypoint_results[0].keypoints[0].numpy()  # Convert to numpy array

            # Apply smoothing to keypoints (only if prev_keypoints exists)
            if prev_keypoints is not None:
                keypoints = smoothing_factor * keypoints + (1 - smoothing_factor) * prev_keypoints

            annotated_frame = keypoint_results[0].plot()

            # Overlay the annotated ROI onto the frame
            frame[max(0, y1 - roi_margin):min(frame.shape[0], y2 + roi_margin),
                  max(0, x1 - roi_margin):min(frame.shape[1], x2 + roi_margin)] = annotated_frame

            prev_keypoints = keypoints  # Store the current keypoints for next frame

        else:
            print("No keypoints detected for the athlete.")

    # Show the frame with detection and keypoints
    cv2.imshow("Athlete Keypoint Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

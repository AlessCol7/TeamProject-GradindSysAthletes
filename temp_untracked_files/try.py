import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Initialize DeepSORT tracker
tracker = DeepSort()

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
seconds_interval = 2  # Track movement for every 2 seconds

# Initialize dictionaries to store movement and proximity data
movement_counts = defaultdict(float)  # Store total movement distance for each tracked person
proximity_counts = defaultdict(float)  # Store proximity to the reference point for each person
time_interval_frames = int(cap.get(cv2.CAP_PROP_FPS) * seconds_interval)  # Frames corresponding to 2 seconds

# Process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Calculate reference_point based on the first frame
    if frame_count == 0:
        reference_point = (frame.shape[1] // 2, frame.shape[0] // 2)  # Center of the frame

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
        detected_people = []  # To store bounding boxes for people detected in the current frame
        if contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out small movements
                if w * h < 500:  # Threshold for movement area (adjust as needed)
                    continue

                # Perform YOLO detection to check if the object is a person
                results = model(frame[y:y+h, x:x+w], save=False)
                for result in results[0].boxes:
                    if result.cls == 0:  # Class 0 is "person"
                        track_id = result.id
                        movement_distance = np.sqrt((x - y) ** 2 + (y - y) ** 2)

                        movement_counts[track_id] += movement_distance

                        # Calculate proximity to the reference point (center of the frame)
                        person_center = (x + w // 2, y + h // 2)
                        proximity_distance = np.sqrt((person_center[0] - reference_point[0]) ** 2 +
                                                     (person_center[1] - reference_point[1]) ** 2)
                        proximity_counts[track_id] += proximity_distance
                        detected_people.append((track_id, x, y, w, h))

    # Update previous frame for motion detection
    previous_frame = gray_frame.copy()

    # If we've processed enough frames (2 seconds), determine the most active and closest person
    if frame_count % time_interval_frames == 0 and detected_people:
        # If only one person is detected, automatically select them
        if len(detected_people) == 1:
            best_person = detected_people[0][0]
        else:
            # Combine movement and proximity scores for each person
            combined_scores = {track_id: movement_counts[track_id] + proximity_counts[track_id]
                               for track_id in movement_counts}

            # Select the person with the highest combined score (most active and closest)
            best_person = max(combined_scores, key=combined_scores.get)
        
        print(f"Selected person ID: {best_person}")

        # Perform keypoint detection only for the selected person
        for track_id, x, y, w, h in detected_people:
            if track_id == best_person:  # Track the selected person only
                roi = frame[y:y+h, x:x+w]
                # Perform keypoint detection only for the selected person
                keypoint_results = model(roi, save=False)
                annotated_frame = keypoint_results[0].plot()

                # Overlay annotated keypoints onto the frame
                frame[y:y+h, x:x+w] = annotated_frame

    # Display the processed frame
    cv2.imshow("Athlete Keypoint Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

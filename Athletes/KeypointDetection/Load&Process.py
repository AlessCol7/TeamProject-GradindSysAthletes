import cv2
import numpy as np
from collections import defaultdict
from mediapipe import solutions as mp_solutions

# Step 1: Identify the most frequent person

def detect_most_frequent_person(video_path):
    cap = cv2.VideoCapture(video_path)
    person_counts = defaultdict(int)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_id = (x // 10, y // 10)  # Simplified face ID (bucketized coordinates)
            person_counts[face_id] += 1
        
        frame_count += 1
        if frame_count > 1000:  # Analyze only the first 1000 frames
            break

    cap.release()
    most_frequent_person = max(person_counts, key=person_counts.get, default=None)
    return most_frequent_person

# Step 2: Apply keypoint detection

def apply_keypoint_detection(video_path, target_person):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp_solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('Keypoint Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
video_path = "YouTubeDownload/DownloadedVideos/atletiek.mp4"
most_frequent_person = detect_most_frequent_person(video_path)
if most_frequent_person:
    apply_keypoint_detection(video_path, most_frequent_person)
else:
    print("No person detected in the video.")

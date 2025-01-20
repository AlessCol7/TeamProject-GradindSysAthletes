import cv2
import numpy as np
import tensorflow as tf
import joblib
import os
from ultralytics import YOLO

# Define available options
activities = {
    "1": {"name": "Discuswerper", "model": "athlete_score_predictor1.h5", "scaler": "scoring_scaler_discuswerper.pkl", "video": "discuswerper_video.mp4"},
    "2": {"name": "Estafette", "model": "athlete_score_predictor2.h5", "scaler": "scoring_scaler_estafette.pkl", "video": "estafette_video.mp4"},
    "3": {"name": "Hoogspringen", "model": "athlete_score_predictor3.h5", "scaler": "scoring_scaler_hoogspringen.pkl", "video": "hoogspringen_video.mp4"},
    # Add other activities as needed
}

# Function to normalize keypoints
def normalize_keypoints(keypoints, bbox, expected_size):
    normalized_keypoints = []
    for i in range(0, len(keypoints), 2):
        x = keypoints[i] / bbox[2]  # Normalize by width
        y = keypoints[i + 1] / bbox[3]  # Normalize by height
        normalized_keypoints.extend([x, y])
    # Pad or trim to match the expected size
    if len(normalized_keypoints) > expected_size:
        normalized_keypoints = normalized_keypoints[:expected_size]
    elif len(normalized_keypoints) < expected_size:
        normalized_keypoints += [0] * (expected_size - len(normalized_keypoints))
    return normalized_keypoints

# Main function to score a video
def score_video(activity):
    model_path = activity["model"]
    scaler_path = activity["scaler"]
    video_path = activity["video"]
    
    # Load the model and scaler
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load YOLO model for pose estimation
    pose_model = YOLO("yolov8n-pose.pt")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_scores = []
    expected_size = 17 * 3  # Adjust based on the expected number of keypoints
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 1: Extract Keypoints
        results = pose_model(frame)
        if len(results) == 0 or results[0].keypoints is None:
            continue
        
        keypoints = results[0].keypoints.xy.cpu().numpy().flatten()
        bbox = results[0].boxes.xyxy.cpu().numpy()[0]
        
        if bbox is None or len(keypoints) == 0:
            continue  # Skip frames without valid bbox or keypoints
        
        # Step 2: Normalize keypoints
        normalized_keypoints = normalize_keypoints(keypoints, bbox, expected_size)
        
        # Step 3: Predict Score
        normalized_keypoints = np.array([normalized_keypoints], dtype=np.float32)
        normalized_keypoints = scaler.transform(normalized_keypoints)
        score = model.predict(normalized_keypoints).flatten()[0]
        score = np.clip(score, 1, 5)  # Ensure the score is between 1 and 5
        rounded_score = round(score * 2) / 2  # Round to the nearest 0.5
        frame_scores.append(rounded_score)
        
        # Step 4: Annotate frame
        cv2.putText(frame, f"Score: {rounded_score:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Video', frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Step 5: Calculate average score
    average_score = np.mean(frame_scores) if frame_scores else 0
    print(f"Average Score for {activity['name']}: {average_score:.2f}")

# Display menu and get user choice
def main():
    print("Select an activity to score a video:")
    for key, activity in activities.items():
        print(f"{key}. {activity['name']}")
    
    choice = input("Enter the number of your choice: ")
    if choice in activities:
        score_video(activities[choice])
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

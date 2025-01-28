import numpy as np
import cv2
import joblib
import tensorflow as tf
from ultralytics import YOLO
from sklearn.preprocessing import MinMaxScaler

# Custom MSE Metric
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

tf.keras.utils.get_custom_objects()["mse"] = mse

# Paths for Scoring Models and Scalers
scoring_models = {
    "Sprint Start": "new_models/Sprint_Start_1.h5",
    "Sprint Running": "new_models/Sprint.h5",
    "Shot Put": "new_models/Kogelstoten.h5",
    "Relay Receiver": "new_models/Estafette.h5",
    "Long Jump": "new_models/Verspringen.h5",
    "Javelin": "new_models/Speerwerpen.h5",
    "High Jump": "new_models/Hoogspringen.h5",
    "Discus Throw": "new_models/Discurweper.h5",
    "Hurdling": "new_models/Hoogspringen.h5"
}

# YOLO Pose Detection Model
pose_model = YOLO("yolov8s-pose.pt")

def extract_keypoints_from_frame(frame, pose_model, expected_size):
    """Extract keypoints and normalize them."""
    results = pose_model(frame)
    if not results or results[0].keypoints is None:
        return None

    keypoints = results[0].keypoints.xy.cpu().numpy().flatten()
    expected_num_keypoints = 17 * 3  # 17 keypoints with x, y, and confidence
    normalized_keypoints = np.pad(
        keypoints, (0, max(0, expected_num_keypoints - len(keypoints))), mode="constant"
    )[:expected_num_keypoints]
    return normalized_keypoints

def score_video(video_path, scoring_model_path, scaler_path, pose_model, expected_size):
    """Score a video based on the chosen exercise."""
    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(scoring_model_path)
    cap = cv2.VideoCapture(video_path)
    frame_scores = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = extract_keypoints_from_frame(frame, pose_model, expected_size)
        if keypoints is None:
            continue
        keypoints_scaled = scaler.transform([keypoints])
        score = model.predict(keypoints_scaled).flatten()[0]
        frame_scores.append(np.clip(score, 1, 5))

    cap.release()
    return np.mean(frame_scores) if frame_scores else 0

def main_menu():
    """Display the menu and handle user input."""
    print("\n--- Exercise Scoring Menu ---")
    print("Choose an exercise to score:")
    for idx, exercise in enumerate(scoring_models.keys(), 1):
        print(f"{idx}. {exercise}")
    print("0. Exit")

    try:
        choice = int(input("\nEnter your choice: "))
        if choice == 0:
            print("Exiting the program.")
            return None, None
        exercise_name = list(scoring_models.keys())[choice - 1]
        video_path = input("Enter the path to the video file: ")
        return exercise_name, video_path
    except (ValueError, IndexError):
        print("Invalid choice. Please try again.")
        return None, None

if __name__ == "__main__":
    expected_size = 51  # Number of features (17 keypoints * 3 values)
    
    while True:
        exercise_name, video_path = main_menu()
        if not exercise_name or not video_path:
            break
        
        try:
            # Get model and scaler paths for the chosen exercise
            scoring_model_path, scaler_path = scoring_models[exercise_name]
            
            # Score the video
            average_score = score_video(video_path, scoring_model_path, scaler_path, pose_model, expected_size)
            print(f"\nAverage Score for {exercise_name}: {average_score:.2f}")
        except Exception as e:
            print(f"An error occurred: {e}")

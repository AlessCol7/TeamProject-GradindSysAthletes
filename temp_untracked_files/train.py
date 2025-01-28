import json
import numpy as np
import os
from collections import Counter
from scipy.special import softmax
import cv2  # OpenCV for video processing
from ultralytics import YOLO  # YOLOv8 library
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the mapping of exercise names to labels
exercise_labels = {
    "Discurweper.json": 0,
    "Estafette.json": 1,
    "Hoogspringen.json": 2,
    "Hordenlopen.json": 3,
    "Kogelstoten.json": 4,
    "Speerwerpen.json": 5,
    "sprint_start.json": 6,
    "sprint.json": 7,
    "Verspringen.json": 8,
}

# Function to load keypoints from JSON files
def load_keypoints(json_folder):
    X, y = [], []
    expected_num_keypoints = 17 * 2  # 17 keypoints with x, y coordinates

    for json_file, label in exercise_labels.items():
        json_path = os.path.join(json_folder, json_file)
        if not os.path.exists(json_path):
            print(f"Warning: {json_file} does not exist in {json_folder}. Skipping.")
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        for segment in data["segments"].values():
            for annotation in segment["annotations"]:
                keypoints = annotation["keypoints"]
                if len(keypoints) < 51:  # 17 keypoints * 3 (x, y, visibility)
                    print(f"Warning: Annotation ID {annotation['id']} in {json_file} has insufficient keypoints. Skipping.")
                    continue

                # Extract x, y coordinates only and flatten
                normalized_keypoints = np.array(keypoints).reshape(-1, 3)[:, :2].flatten()

                # Pad or truncate to ensure consistent length
                if len(normalized_keypoints) < expected_num_keypoints:
                    normalized_keypoints = np.pad(
                        normalized_keypoints,
                        (0, expected_num_keypoints - len(normalized_keypoints)),
                        mode="constant"
                    )
                elif len(normalized_keypoints) > expected_num_keypoints:
                    normalized_keypoints = normalized_keypoints[:expected_num_keypoints]

                X.append(normalized_keypoints)
                y.append(label)

    return np.array(X), np.array(y)

# Load JSON keypoints
json_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/JsonScore"
X, y = load_keypoints(json_folder)

print(f"Feature matrix shape: {X.shape}")
print(f"Labels array shape: {y.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = clf.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Reverse mapping of exercise labels
exercise_labels_inv = {v: k for k, v in exercise_labels.items()}

# YOLO Keypoints extraction and prediction function
def extract_keypoints_and_predict(frame, model, clf, scaler):
    results = model(frame)
    if not results:
        raise ValueError("No detections made in the frame.")

    # Extract keypoints
    keypoints_list = []
    for result in results:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # x, y coordinates
            keypoints_list.append(keypoints)

    if not keypoints_list:
        raise ValueError("No keypoints detected in frame.")

    # Use the first detected pose
    keypoints = keypoints_list[0]
    keypoints_flattened = keypoints.flatten()

    if keypoints_flattened.size != scaler.n_features_in_:
        print(f"Expected {scaler.n_features_in_} features, got {keypoints_flattened.size}. Skipping.")
        return None, None

    # Normalize and predict
    scaled_keypoints = scaler.transform([keypoints_flattened])
    probabilities = clf.predict_proba(scaled_keypoints)[0]
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities



# Load the YOLOv8 pose model
model = YOLO("yolov8s-pose.pt")  # Replace with your model path

# Assuming clf, scaler, and exercise_labels_inv are pre-defined
# clf: The classifier used for exercise classification
# scaler: Scaler used to preprocess keypoints before classification
# exercise_labels_inv: Mapping from class indices to class names

def extract_keypoints_and_predict(frame, model, clf, scaler):
    """
    Process a frame, detect keypoints, and predict exercise class and probabilities.
    """
    # Detect keypoints using YOLO model
    results = model(frame)  # YOLO inference
    if not results:
        raise ValueError("No detections made in the frame.")
    
    # Extract keypoints from the results
    keypoints_list = []
    for result in results:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            # Extract raw keypoint data (xy coordinates)
            keypoints = result.keypoints.xy.cpu().numpy()  # Ensure NumPy array on CPU
            keypoints_list.append(keypoints)
    
    if not keypoints_list:
        raise ValueError("No keypoints detected in frame.")
    
    # Use the first set of keypoints (or implement logic to handle multiple detected poses)
    keypoints = keypoints_list[0]  # Assuming only one person per frame

    # Preprocess keypoints (scaling and reshaping for the classifier)
    keypoints_flattened = keypoints.flatten()  # Flatten to a 1D array
    scaled_keypoints = scaler.transform([keypoints_flattened])  # Scale and reshape
    probabilities = clf.predict_proba(scaled_keypoints)[0]
    predicted_class = np.argmax(probabilities)
    return predicted_class, probabilities

# Path to the video
video_path = '/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/KeypointDetection/exercises/Discurweper/segment_001557.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
class_counts = Counter()
class_probabilities = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    print(f"Processing frame {frame_count + 1}")
    try:
        # Resize frame
        frame_resized = cv2.resize(frame, (640, 640))
        
        # YOLO inference
        results = model(frame_resized)
        print(f"Frame {frame_count + 1}: {len(results)} objects detected.")

        # Extract keypoints and classify
        exercise_class, exercise_class_proba = extract_keypoints_and_predict(frame_resized, model, clf, scaler)
        if exercise_class is not None:
            print(f"Frame {frame_count + 1}: Classified as {exercise_labels_inv[exercise_class]} with confidence {np.max(exercise_class_proba) * 100:.2f}%")
            frame_count += 1
            class_counts[exercise_class] += 1
            class_probabilities.append(exercise_class_proba)
    except ValueError as e:
        print(f"Skipping frame {frame_count + 1} due to error: {e}")
        continue

cap.release()


# Calculate overall percentages based on frame count
class_labels = clf.classes_
total_frames = sum(class_counts.values())

if total_frames == 0:
    print("No frames were classified.")
else:
    class_percentages = {exercise_labels_inv[c]: (count / total_frames) * 100 for c, count in class_counts.items()}
    
    # Calculate average probabilities across all frames
    if class_probabilities:
        average_probabilities = np.mean(class_probabilities, axis=0)
        probability_percentages = {exercise_labels_inv[i]: p * 100 for i, p in enumerate(average_probabilities)}
    else:
        probability_percentages = {}

    # Output results
    print("Class Percentages (Based on Frame Count):")
    for cls, pct in class_percentages.items():
        print(f"{cls}: {pct:.2f}%")
    
    print("\nAverage Probabilities (Softmax Scores):")
    for cls, pct in probability_percentages.items():
        print(f"{cls}: {pct:.2f}%")
def round_to_half(number):
    """
    Round a number to the nearest integer or .5.
    """
    return round(number * 2) / 2

# Predicting the weighted score based on the average probabilities
weighted_probs = np.multiply(average_probabilities, np.arange(len(class_labels)))
predicted_score = np.sum(weighted_probs) / np.sum(average_probabilities)
rounded_predicted_score = round_to_half(predicted_score)
print(f"\nWeighted Predicted Score: {rounded_predicted_score:.2f}")

    # # Predicting the weighted score based on the average probabilities
    # weighted_probs = np.multiply(average_probabilities, np.arange(len(class_labels)))
    # predicted_score = np.sum(weighted_probs) / np.sum(average_probabilities)
    # print(f"\nWeighted Predicted Score: {predicted_score:.2f}")

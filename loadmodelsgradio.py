# This one giving different scores!!!!!!!

import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.decomposition import PCA

# Define the mapping between sport branches and models
SPORT_BRANCH_MODEL_MAPPING = {
    "Sprint Start": "new_models/Sprint_Start.h5",
    "Sprint Running": "new_models/Sprint.h5",
    "Shot Put": "new_models/Kogelstoten.h5",
    "Relay Receiver": "new_models/Estafette.h5",
    "Long Jump": "new_models/Verspringen.h5",
    "Javelin": "new_models/Speerwerpen.h5",
    "High Jump": "new_models/Hoogspringen.h5",
    "Discus Throw": "new_models/Discurweper.h5",
    "Hurdling": "new_models/Hoogspringen.h5",
}

# Load MobileNetV2 for feature extraction
mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
TARGET_FEATURE_SIZE = 51

def load_model(sport_branch):
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    if not model_path:
        return None, f"Error: No model found for sport branch: {sport_branch}"
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


# def extract_features_from_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None, "Error: Couldn't read video stream from file"

#     features = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Resize and preprocess the frame
#         frame_resized = cv2.resize(frame, (224, 224))
#         frame_preprocessed = preprocess_input(frame_resized)

#         # Extract features using MobileNetV2
#         feature_map = mobilenet_model.predict(np.expand_dims(frame_preprocessed, axis=0))
#         feature_vector = feature_map.flatten()
#         features.append(feature_vector)

#     cap.release()

#     if not features:
#         return None, "Error: No frames extracted from video."

#     # Average the features over all frames
#     features = np.array(features)
#     mean_features = np.mean(features, axis=0)

#     print(f"Extracted features shape: {features.shape}")
#     print(f"Mean features shape: {mean_features.shape}")

#     return mean_features, None
def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Couldn't read video stream from file"

    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess the frame
        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(frame_resized)

        # Extract features using MobileNetV2
        feature_map = mobilenet_model.predict(np.expand_dims(frame_preprocessed, axis=0))
        feature_vector = feature_map.flatten()
        features.append(feature_vector)

    cap.release()

    if not features:
        return None, "Error: No frames extracted from video."

    # Average the features over all frames
    features = np.array(features)
    mean_features = np.mean(features, axis=0)

    # Reshape or truncate features to match the model input
    target_shape = 51
    if len(mean_features) > target_shape:
        mean_features = mean_features[:target_shape]  # Truncate
    else:
        mean_features = np.pad(mean_features, (0, target_shape - len(mean_features)), mode='constant')  # Pad with zeros

    mean_features = np.reshape(mean_features, (1, 1, target_shape))
    print(f"Reshaped features: {mean_features.shape}")

    return mean_features, None


def make_prediction(model, video_path):
    features, error = extract_features_from_video(video_path)
    if error:
        return error

    features = np.reshape(features, (1, -1))  # Ensure the shape matches the model's input
    predictions = model.predict(features)
    print(f"Predictions: {predictions}")

    average_score = np.mean(predictions.flatten())
    average_score = np.clip(average_score, 0, 5)
    return f"Average Score: {average_score:.2f} out of 5"

def predict(video, sport_branch):
    if isinstance(video, str):
        # Video is a file path, use it directly
        video_path = video
    else:
        # Video is a file-like object, write it to a temporary file
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video.read())  # Write file content to disk

    # Proceed with video processing
    print(f"Processing video from: {video_path}")

    # Load the model for the sport branch
    model, error = load_model(sport_branch)
    if error:
        return error

    # Extract features and make predictions
    result = make_prediction(model, video_path)
    return result


# Gradio Interface
sport_options = list(SPORT_BRANCH_MODEL_MAPPING.keys())
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Dropdown(sport_options, label="Select Sport Branch")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Sport Performance Evaluation",
    description="Upload a video and select the corresponding sport branch to evaluate performance."
)

if __name__ == "__main__":
    interface.launch()
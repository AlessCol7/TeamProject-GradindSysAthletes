import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile
import os

# Mapping of sport branches to their corresponding models
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

sport_options = list(SPORT_BRANCH_MODEL_MAPPING.keys())

# Load MobileNetV2 for feature extraction
mobilenet_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
TARGET_FEATURE_SIZE = 51

def load_model(sport_branch):
    """
    Load the appropriate model for the selected sport branch.
    """
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    print(f"Loading model from: {model_path}")
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    if not model_path:
        return None, f"No model found for the sport branch: {sport_branch}"
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def extract_features_from_video(video_path):
    """
    Extracts features from the video using MobileNetV2.
    """
    mean_features = np.mean(features, axis=0)
    print(f"Mean features shape: {mean_features.shape}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file."

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
        return None, "No frames extracted from the video."

    # Average the features over all frames
    features = np.array(features)
    mean_features = np.mean(features, axis=0)

    # Adjust features to match the model input
    if len(mean_features) > TARGET_FEATURE_SIZE:
        mean_features = mean_features[:TARGET_FEATURE_SIZE]  # Truncate
    else:
        mean_features = np.pad(mean_features, (0, TARGET_FEATURE_SIZE - len(mean_features)), mode="constant")  # Pad with zeros

    mean_features = mean_features.reshape((1, TARGET_FEATURE_SIZE))
    return mean_features, None


def make_prediction(model, video_path):
    features, error = extract_features_from_video(video_path)
    if error:
        return error
    print(f"Features fed to model: {features.shape}")
    predictions = model.predict(features)
    print(f"Predictions: {predictions}")
    features, error = extract_features_from_video(video_path)
    if error:
        return error

    try:
        predictions = model.predict(features)
        average_score = np.mean(predictions.flatten())
        average_score = np.clip(average_score, 0, 5)  # Limit score to 0-5 range
        return f"Average Score: {average_score:.2f} out of 5"
    except Exception as e:
        return f"Prediction error: {str(e)}"


def predict(video, sport_branch):
    """
    Main function to handle video uploads and prediction.
    """
    print(f"Video input: {video}")
    if isinstance(video, str):
        # If video is already a file path
        video_path = video
    else:
        # Handle the case where video is uploaded differently
        return "Invalid input. Expected a video file path."

    # Load the appropriate model
    model, error = load_model(sport_branch)
    if error:
        return error

    # Make predictions using the model
    return make_prediction(model, video_path)


# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Video(label="Upload Video"),  # Ensure compatibility
        gr.Dropdown(list(SPORT_BRANCH_MODEL_MAPPING.keys()), label="Select Sport Branch"),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Sport Performance Evaluation",
    description="Upload a video and select the sport branch to evaluate the performance.",
)


if __name__ == "__main__":
    interface.launch()

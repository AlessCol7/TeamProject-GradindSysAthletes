# best and final code
import gradio as gr
import pyodbc
import hashlib
import requests
import os
from settings.config import CONNECTION_STRING, TRIGGER_URL
import re
from datetime import datetime
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.decomposition import PCA
import pandas as pd
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


# Validate email format
def is_valid_student_email(email):
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@student\.howest\.be$', email))

def is_valid_teacher_email(email):
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@howest\.be$', email))

def validate_email(email, role):
    if role == 'student' and not is_valid_student_email(email):
        return "Error: Invalid student email format. It should be name.lastname@student.howest.be"
    elif role == 'teacher' and not is_valid_teacher_email(email):
        return "Error: Invalid teacher email format. It should be name.lastname@howest.be"
    return None

def validate_password(password):
    if len(password) < 8:
        return "Error: Password must be at least 8 characters long."
    if ' ' in password:
        return "Error: Password should not contain spaces."
    return None

def connect_to_db():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=tcp:atish.database.windows.net,1433;'
                      'Database=atish-LoginData;'
                      'Uid=atish;Pwd=13sql17_ctai;'
                      'Encrypt=yes;TrustServerCertificate=no;'
                      'Connection Timeout=30;')

    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(first_name, last_name, email, password, role):
    email_error = validate_email(email, role)
    if email_error:
        return email_error

    password_error = validate_password(password)
    if password_error:
        return password_error

    hashed_password = hash_password(password)
    conn = connect_to_db()
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM Users WHERE Email = ?", (email,))
    if cursor.fetchone():
        return "Error: Email already exists."

    cursor.execute("""
        INSERT INTO Users (FirstName, LastName, Email, Password, Role)
        VALUES (?, ?, ?, ?, ?)
    """, (first_name, last_name, email, hashed_password, role))
    conn.commit()
    return f"User '{first_name} {last_name}' registered successfully as {role}."


def validate_login(email, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Password, Role FROM Users WHERE Email = ?", (email,))
    user_data = cursor.fetchone()

    if user_data:
        stored_password, role = user_data
        input_password_hashed = hash_password(password)
        if stored_password == input_password_hashed:
            return role
    return "Invalid email or password"

def student_view():
    return "Welcome, Student! You can now upload videos for evaluation."

def teacher_view():
    return "Welcome, Teacher! You can view all students' videos and results."

def login_page(email, password):
    role = validate_login(email, password)
    if role == "student":
        global current_user_email
        current_user_email = email
        return student_view()
    elif role == "teacher":
        current_user_email = email
        return teacher_view()
    else:
        return "Invalid email or password. Please try again."

def register_page(first_name, last_name, email, password, role):
    if not first_name.strip():
        return "Error: First name is required."
    if not last_name.strip():
        return "Error: Last name is required."
    
    registration_result = register_user(first_name, last_name, email, password, role)
    
    if "Error" in registration_result:
        return registration_result
    
    global current_user_email
    current_user_email = email
    
    if role == 'student':
        return student_view()
    elif role == 'teacher':
        return teacher_view()

UPLOAD_API_URL = TRIGGER_URL
current_user_email = None

def trim_video(input_path, output_path, threshold=5, frame_skip=5):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Error: Couldn't read video stream from file."

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer for trimmed output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    prev_frame = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames for faster processing
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray_frame)
            mean_diff = np.mean(diff)

            if mean_diff > threshold: # Only write frames with significant activity
                out.write(frame)

        prev_frame = gray_frame
    cap.release()
    out.release()
    return output_path, None


def upload_video(file, sport_branch, guess_grade=None):
    if current_user_email is None:
        return "You must log in before uploading a video."
    if file is None:
        return "Please upload a video file."
    if sport_branch not in SPORT_BRANCH_MODEL_MAPPING.keys():
        return "Invalid sport branch selection."
    if isinstance(file, str):
        file_path = file
    else:
        file_path = "temp_video.mp4"
        with open(file_path, "wb") as f:
            f.write(file.read())

    with open(file_path, "rb") as video_file:
        files = {"file": video_file}
        try:
            response = requests.post(TRIGGER_URL, files=files)
            if response.status_code != 200:
                return f"Error: Failed to upload video. Status code: {response.status_code}"
            
            response_data = response.json()
            print("API Response Data:", response_data)  # Debugging: Log the response data
            
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to the API. Details: {str(e)}"
        except ValueError:
            return "Error: Received invalid JSON response from the API."

    # Step 2: Check if the response contains a video URL
    video_url = response_data.get("fileUrl")
    if not video_url:
        print("Missing 'fileUrl' in response:", response_data)
        error_message = response_data.get("error", "Unknown error")
        return f"Error: Could not retrieve video URL from the response. Details: {error_message or response.text}"

    print(f"Video uploaded to: {video_url}")

    # Load model for the sport branch
    model = load_model(sport_branch)
    if isinstance(model, str):
        return model

    # Step 3: Make prediction on the uploaded video
    prediction_result = make_prediction(model, file_path)
    print(f"Prediction result for {sport_branch}: {prediction_result}")

    if prediction_result.startswith("Error"): # save video details into database
        return prediction_result

    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (current_user_email,))
    user_data = cursor.fetchone()
    if user_data:
        user_id = user_data[0]
        cursor.execute("""
            INSERT INTO Videos (UserID, FileName, UploadTime, SportBranch, Score, GuessGrade)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, video_url, datetime.now(), sport_branch, prediction_result, guess_grade))
        conn.commit()

    return f"Video uploaded successfully for {sport_branch}! {prediction_result}"


SPORT_BRANCH_MODEL_MAPPING = {
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

# Load pre-trained model for feature extraction (MobileNetV2)
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

pose_model = YOLO("yolov8n-pose.pt")

TARGET_FEATURE_SIZE = 51

scaler_path = "scoring_scaler7.pkl"
if os.path.exists(scaler_path):
    scaler_scoring = joblib.load(scaler_path)
else:
    scaler_scoring = MinMaxScaler(feature_range=(0, 1))  # Placeholder; training data required to fit


def load_model(sport_branch):
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    if not model_path:
        return f"Error: No model found for sport branch: {sport_branch}"

    print(f"Loading model for {sport_branch} from: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        return model
    except Exception as e:
        return f"Error loading model for {sport_branch}: {str(e)}"


def inspect_model_weights(model):
    for layer in model.layers:
        weights = layer.get_weights()
        print(f"Weights for layer {layer.name}: {weights}")


def test_model_with_synthetic_data(model):
    synthetic_features = np.random.random((1, 51))
    synthetic_prediction = model.predict(synthetic_features)
    print(f"Prediction on synthetic input: {synthetic_prediction}")


def check_input_features(features):
    print(f"Input features shape: {features.shape}")
    print(f"Sample input features (first 5 rows):\n{features[:5]}")
    print(f"Standard deviation of features: {np.std(features, axis=0).mean()}")


def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    print(f"Normalized features (first 5 rows):\n{normalized_features[:5]}")
    return normalized_features


def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Couldn't read video stream from file"

    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame)
        if len(results) == 0 or results[0].keypoints is None:
            continue

        keypoints = results[0].keypoints.xy.cpu().numpy().flatten()
        if len(keypoints) == 0:
            continue

        bbox = results[0].boxes.xyxy.cpu().numpy()[0]
        if bbox is None or len(bbox) != 4:
            continue

        normalized_keypoints = []
        for i in range(0, len(keypoints), 2):
            x = keypoints[i] / bbox[2]
            y = keypoints[i + 1] / bbox[3]
            normalized_keypoints.extend([x, y])

        if len(normalized_keypoints) > TARGET_FEATURE_SIZE:
            normalized_keypoints = normalized_keypoints[:TARGET_FEATURE_SIZE]
        elif len(normalized_keypoints) < TARGET_FEATURE_SIZE:
            normalized_keypoints += [0] * (TARGET_FEATURE_SIZE - len(normalized_keypoints))

        features.append(normalized_keypoints)

    cap.release()

    if not features:
        return None, "Error: No frames extracted from video."

    features = np.array(features)
    mean_features = np.mean(features, axis=0)
    mean_features = np.reshape(mean_features, (1, -1))

    return mean_features, None


def make_prediction(model, video_path):
    test_features = np.random.random((1, 51))
    print(f"Testing with random input: {test_features}")
    test_predictions = model.predict(test_features)
    print(f"Test prediction on random input: {test_predictions}")

    features, error = extract_features_from_video(video_path)
    if error:
        return error

    if features.size == 0:
        return "Error: No frames extracted from video."

    normalized_features = scaler_scoring.transform(features)
    predictions = model.predict(normalized_features)
    frame_scores = predictions.flatten()
    average_score = np.mean(frame_scores)
    print(f"Average score before clipping: {average_score}")
    average_score = np.clip(average_score, 0, 5)
    return f"Average Score: {average_score:.2f} out of 5"


def get_uploaded_videos(email, role):
    conn = connect_to_db()
    cursor = conn.cursor()

    if role == "student": # Fetch results based on role
        query = """
            SELECT v.FileName, v.SportBranch, v.UploadTime, u.Email, v.Score, v.GuessGrade
            FROM Videos v
            INNER JOIN Users u ON v.UserID = u.UserID
            WHERE u.Email = ?
        """
        cursor.execute(query, (email,))
    elif role == "teacher":
        query = """
            SELECT v.FileName, v.SportBranch, v.UploadTime, u.Email, v.Score, v.GuessGrade
            FROM Videos v
            INNER JOIN Users u ON v.UserID = u.UserID
        """
        cursor.execute(query)
    else:
        return "Error: Unauthorized access."

    uploaded_videos = cursor.fetchall()

    # Create the table header
    table_html = """
    <table style="width:100%; border-collapse: collapse; text-align: left;">
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">Video</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Sport</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Uploaded by</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Time</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Actual Grade</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Guessed Grade</th>
        </tr>
    """

    # Populate the table with data
    for i, video in enumerate(uploaded_videos):
        video_path = video[0].split("/")[-1]  # Show only the file name
        sport_branch = video[1]
        upload_time = video[2].strftime("%Y-%m-%d %H:%M")  # Simplify time format
        email = video[3]
        score = f"{video[4]} / 5"  # Simplify score display
        guessed_grade = f"{video[5]} / 5" if video[5] is not None else "N/A"
        
        row_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"  # Alternate row colors

        table_html += f"""
        <tr style="background-color: {row_color};">
            <td style="border: 1px solid #ddd; padding: 8px;" title="{video_path}">{video_path[:20]}...</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{sport_branch}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{email}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{upload_time}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{score}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{guessed_grade}</td>
        </tr>
        """

    table_html += "</table>"
    return table_html


with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Welcome to the Athletics Evaluation System 🏃‍♂️🏅")

    with gr.Tab("Register"):
        first_name_input = gr.Textbox(label="First Name *")
        last_name_input = gr.Textbox(label="Last Name *")
        email_input = gr.Textbox(label="Email *")
        password_input_reg = gr.Textbox(label="Password *", type="password")
        role_input_reg = gr.Radio(["student", "teacher"], label="Role *")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Result", interactive=False)

        register_btn.click(
            register_page,
            inputs=[first_name_input, last_name_input, email_input, password_input_reg, role_input_reg],
            outputs=register_output
        )

    with gr.Tab("Login"):
        email_input_log = gr.Textbox(label="Email *")
        password_input_log = gr.Textbox(label="Password *", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)

        def validate_login_fields(email, password):
            if not email or not password:
                return "Error: All fields are mandatory. Please fill out all required fields."
            return login_page(email, password)

        login_btn.click(
            validate_login_fields,
            inputs=[email_input_log, password_input_log],
            outputs=login_output
        )

    with gr.Tab("Upload Video"):
        sport_branch_input = gr.Dropdown(
            ["", "Sprint Start", "Sprint Running", "Shot Put", "Relay Receiver", "Long Jump", "Javelin", "High Jump", "Discus Throw", "Hurdling"],
            label="Select Sport Branch *"
        )
        video_input = gr.Video(label="Upload Video *")
        guess_grade_input = gr.Slider(
            minimum=0, maximum=5, step=0.1, label="Guess Grade (0 to 5)", interactive=True
        )
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")

        def validate_upload_fields(file, sport_branch, guess_grade):
            if current_user_email is None:
                return "You must log in before uploading a video."
            if not file:
                return "Error: Please upload a video file."
            if not sport_branch or sport_branch == "":
                return "Error: Please select a sport branch."

            # Check role
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT Role FROM Users WHERE Email = ?", (current_user_email,))
            user_role = cursor.fetchone()

            if user_role and user_role[0] == "student" and guess_grade is None:
                return "Error: Guess Grade is mandatory for students."
            
            return upload_video(file, sport_branch, guess_grade)

        upload_btn.click(
            validate_upload_fields,
            inputs=[video_input, sport_branch_input, guess_grade_input],
            outputs=upload_output
        )

    with gr.Tab("View Results"):
        results_output = gr.HTML(label="Video Results")
        get_results_btn = gr.Button("Get Results")

        def display_results():
            if current_user_email is None:
                return "Error: Please log in to view results."

            # Determine role of the current user
            conn = connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT Role FROM Users WHERE Email = ?", (current_user_email,))
            user_role = cursor.fetchone()

            if user_role:
                role = user_role[0]
                results = get_uploaded_videos(current_user_email, role)
                return results
            else:
                return "Error: Unable to determine user role."

        get_results_btn.click(display_results, outputs=results_output)



if __name__ == '__main__':
    #athletics_app.launch(debug=True)
    athletics_app.launch(server_name="0.0.0.0", server_port=7860, share=True)
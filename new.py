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

def upload_video(file, sport_branch):
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

    # Load model for the sport branch
    model = load_model(sport_branch)
    if isinstance(model, str):
        return model

    # Make prediction on uploaded video
    prediction_result = make_prediction(model, file_path)

    # Save video details in the database
    if prediction_result.startswith("Error"):
        return prediction_result

    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (current_user_email,))
    user_data = cursor.fetchone()
    if user_data:
        user_id = user_data[0]
        cursor.execute("""
            INSERT INTO Videos (UserID, FileName, UploadTime, SportBranch, Score)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, file_path, datetime.now(), sport_branch, prediction_result))
        conn.commit()

    return f"Video uploaded successfully for {sport_branch}! {prediction_result}"

SPORT_BRANCH_MODEL_MAPPING = {
    "Sprint Start": "models/Sprint_Start.h5",
    "Sprint Running": "models/Sprint.h5",
    "Shot Put": "models/Kogelstoten.h5",
    "Relay Receiver": "models/Estafette.h5",
    "Long Jump": "models/Verspringen.h5",
    "Javelin": "models/Speerwerpen.h5",
    "High Jump": "models/Hoogspringen.h5",
    "Discus Throw": "models/Discurweper.h5",
    "Hurdling": "models/Hordelopen.h5"
}

# Load pre-trained model for feature extraction (MobileNetV2)
mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def load_model(sport_branch):
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    if not model_path:
        return f"Error: No model found for sport branch: {sport_branch}"
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
        return model
    except Exception as e:
        return f"Error loading model for {sport_branch}: {str(e)}"

TARGET_FEATURE_SIZE = 51

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Couldn't read video stream from file"

    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame to 224x224 and preprocess it for MobileNetV2
        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(frame_resized)

        # Extract features from the frame using MobileNetV2
        feature_map = mobilenet_model.predict(np.expand_dims(frame_preprocessed, axis=0))
        feature_vector = feature_map.flatten()
        
        features.append(feature_vector)

    cap.release()

    # If there are no frames, return an error
    if not features:
        return None, "Error: No frames extracted from video."

    # Stack features and apply PCA to reduce dimensions
    features = np.array(features)
    
    # Apply PCA to reduce the feature dimensions to the desired size
    pca = PCA(n_components=TARGET_FEATURE_SIZE)
    features_reduced = pca.fit_transform(features)

    # Return the average feature vector for the video
    return np.mean(features_reduced, axis=0), None

def make_prediction(model, video_path):
    features, error = extract_features_from_video(video_path)
    if error:
        return error
    
    if features.size == 0:
        return "Error: No frames extracted from video."

    # Reshape to match the model's input shape (1, 51)
    features = np.reshape(features, (1, -1))  # Ensure shape is (1, 51)
    
    # Make the prediction
    predictions = model.predict(features)
    
    # Ensure predictions are meaningful and in the expected range
    frame_scores = predictions.flatten()
    average_score = np.mean(frame_scores)
    average_score = np.clip(average_score, 0, 5)
    
    return f"Average Score: {average_score:.2f} out of 5"   


def get_uploaded_videos():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.FileName, v.SportBranch, v.UploadTime, u.Email, v.Score
        FROM Videos v
        INNER JOIN Users u ON v.UserID = u.UserID
    """)
    uploaded_videos = cursor.fetchall()

    # Create the table header
    table_html = """
    <table style="width:100%; border-collapse: collapse; text-align: left;">
        <tr style="background-color: #f2f2f2;">
            <th style="border: 1px solid #ddd; padding: 8px;">Video</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Sport</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Uploaded by</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Time</th>
            <th style="border: 1px solid #ddd; padding: 8px;">Score</th>
        </tr>
    """

    # Populate the table with data
    for i, video in enumerate(uploaded_videos):
        video_path = video[0].split("/")[-1]  # Show only the file name
        sport_branch = video[1]
        upload_time = video[2].strftime("%Y-%m-%d %H:%M")  # Simplify time format
        email = video[3]
        score = f"{video[4]} / 5"  # Simplify score display
        
        row_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"  # Alternate row colors

        table_html += f"""
        <tr style="background-color: {row_color};">
            <td style="border: 1px solid #ddd; padding: 8px;" title="{video_path}">{video_path[:20]}...</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{sport_branch}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{email}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{upload_time}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{score}</td>
        </tr>
        """

    table_html += "</table>"
    return table_html



with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Welcome to the Athletics Evaluation System")

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
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")

        def validate_upload_fields(file, sport_branch):
            if not file:
                return "Error: Please upload a video file."
            if not sport_branch or sport_branch == "":
                return "Error: Please select a sport branch."
            return upload_video(file, sport_branch)

        upload_btn.click(
            validate_upload_fields,
            inputs=[video_input, sport_branch_input],
            outputs=upload_output
        )

    with gr.Tab("View Results"):
        results_output = gr.HTML(label="Video Results")

        def display_results():
            results = get_uploaded_videos()
            return results

        gr.Button("Get Results").click(display_results, outputs=results_output)

athletics_app.launch(debug=True)

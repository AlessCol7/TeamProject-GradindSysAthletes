import gradio as gr
import pyodbc
import hashlib
import os
import re
from datetime import datetime
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.decomposition import PCA

def is_valid_student_email(email):
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@student\.howest\.be$', email))

def is_valid_teacher_email(email):
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@howest\.be$', email))

def validate_email(email, role):
    if role == 'student' and not is_valid_student_email(email):
        return "Error: Invalid student email format."
    elif role == 'teacher' and not is_valid_teacher_email(email):
        return "Error: Invalid teacher email format."
    return None

def validate_password(password):
    if len(password) < 8:
        return "Error: Password must be at least 8 characters long."
    if ' ' in password:
        return "Error: Password should not contain spaces."
    return None

def connect_to_db():
    return pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                          'Server=tcp:atish.database.windows.net,1433;'
                          'Database=atish-LoginData;'
                          'Uid=atish;Pwd=13sql17_ctai;'
                          'Encrypt=yes;TrustServerCertificate=no;'
                          'Connection Timeout=30;')

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

    cursor.execute(
        """
        INSERT INTO Users (FirstName, LastName, Email, Password, Role)
        VALUES (?, ?, ?, ?, ?)
        """,
        (first_name, last_name, email, hashed_password, role)
    )
    conn.commit()
    return f"User '{first_name} {last_name}' registered successfully as {role}."

def validate_login(email, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Password, Role FROM Users WHERE Email = ?", (email,))
    user_data = cursor.fetchone()

    if user_data:
        stored_password, role = user_data
        if stored_password == hash_password(password):
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

def upload_video(file, sport_branch):
    if current_user_email is None:
        return "You must log in before uploading a video."

    if file is None:
        return "Please upload a video file."

    file_path = "temp_video.mp4"
    with open(file_path, "wb") as f:
        f.write(file)  # Directly write the NamedString object

    model = load_model(sport_branch)
    if isinstance(model, str):
        return model

    prediction_result = make_prediction(model, file_path)



    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (current_user_email,))
    user_data = cursor.fetchone()
    if user_data:
        user_id = user_data[0]
        cursor.execute(
            """
            INSERT INTO Videos (UserID, FileName, UploadTime, SportBranch, Score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, file_path, datetime.now(), sport_branch, prediction_result)
        )
        conn.commit()

    return f"Video uploaded successfully for {sport_branch}! {prediction_result}"

SPORT_BRANCH_MODEL_MAPPING = {
    "Sprint Start": "models/Sprint_Start.h5",
    "Sprint Running": "models/Sprint.h5",
}

mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def load_model(sport_branch):
    model_path = SPORT_BRANCH_MODEL_MAPPING.get(sport_branch)
    if not model_path:
        return f"Error: No model found for sport branch: {sport_branch}"

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        return f"Error loading model for {sport_branch}: {str(e)}"

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Couldn't read video stream."

    features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        frame_preprocessed = preprocess_input(frame_resized)
        feature_map = mobilenet_model.predict(np.expand_dims(frame_preprocessed, axis=0))
        features.append(feature_map.flatten())

    cap.release()
    if not features:
        return None, "Error: No frames extracted."

    features = np.array(features)
    pca = PCA(n_components=51)
    return np.mean(pca.fit_transform(features), axis=0), None

def make_prediction(model, video_path):
    features, error = extract_features_from_video(video_path)
    if error:
        return error

    features = np.reshape(features, (1, -1))
    predictions = model.predict(features)
    return f"Average Score: {np.mean(predictions):.2f} out of 5"

def get_uploaded_videos():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT FileName, SportBranch, UploadTime, Score FROM Videos")
    rows = cursor.fetchall()

    table = "<table>"
    for row in rows:
        table += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td></tr>"
    table += "</table>"
    return table

current_user_email = None

with gr.Blocks() as app:
    with gr.Tab("Login"):
        email = gr.Textbox(label="Email")
        password = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        output = gr.Textbox()
        login_button.click(login_page, inputs=[email, password], outputs=output)

    with gr.Tab("Upload Video"):
        video = gr.File()
        sport_branch = gr.Dropdown(list(SPORT_BRANCH_MODEL_MAPPING.keys()))
        upload_button = gr.Button("Upload")
        upload_output = gr.Textbox()
        upload_button.click(upload_video, inputs=[video, sport_branch], outputs=upload_output)

    with gr.Tab("View Results"):
        view_button = gr.Button("View Uploaded Videos")
        results_output = gr.HTML()
        view_button.click(get_uploaded_videos, outputs=results_output)

app.launch(debug=True)

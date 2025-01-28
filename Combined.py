import gradio as gr
import pyodbc
import hashlib
import requests
import os
from settings.config import CONNECTION_STRING, TRIGGER_URL
import re
from datetime import datetime


# Validate email format
def is_valid_student_email(email):
    # name.lastname@student.howest.be
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@student\.howest\.be$', email))

def is_valid_teacher_email(email):
    # name.lastname@howest.be
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA]+)*@howest\.be$', email))

# Validate email format based on role
def validate_email(email, role):
    if role == 'student':
        if not is_valid_student_email(email):
            return "Error: Invalid student email format. It should be name.lastname@student.howest.be"
    elif role == 'teacher':
        if not is_valid_teacher_email(email):
            return "Error: Invalid teacher email format. It should be name.lastname@howest.be"
    return None  # No errors, email is valid


# Validate password
def validate_password(password):
    # Check if password is at least 8 characters long
    if len(password) < 8:
        return "Error: Password must be at least 8 characters long."
    
    # Check if password contains spaces
    if ' ' in password:
        return "Error: Password should not contain spaces."
    
    return None  # No errors, password is valid


def connect_to_db():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=tcp:atish.database.windows.net,1433;'
                      'Database=atish-LoginData;'
                      'Uid=atish;Pwd=13sql17_ctai;'
                      'Encrypt=yes;TrustServerCertificate=no;'
                      'Connection Timeout=30;')

    return conn
# def connect_to_db():
#     try:
#         conn = pyodbc.connect(CONNECTION_STRING)
#         return conn
#     except pyodbc.Error as e:
#         print(f"Database connection failed: {e}")
#         raise

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Updated register_user function with email validation
def register_user(first_name, last_name, email, password, role):
    # Validate email
    email_error = validate_email(email, role)
    if email_error:
        return email_error
    
    # Validate password
    password_error = validate_password(password)
    if password_error:
        return password_error
    
    # Hash password
    hashed_password = hash_password(password)
    conn = connect_to_db()
    cursor = conn.cursor()

    # Check if email or username already exists
    cursor.execute("SELECT 1 FROM Users WHERE Email = ?", (email,))
    if cursor.fetchone():
        return "Error: Email already exists."

    # Insert the new user into the database
    cursor.execute("""
        INSERT INTO Users (FirstName, LastName, Email, Password, Role)
        VALUES (?, ?, ?, ?, ?)
    """, (first_name, last_name, email, hashed_password, role))
    conn.commit()
    return f"User '{first_name} {last_name}' registered successfully as {role}."

# Function to validate login credentials
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

# Function to display the student view
def student_view():
    return "Welcome, Student! You can now upload videos for evaluation."

# Function to display the teacher view
def teacher_view():
    return "Welcome, Teacher! You can view all students' videos and results."

# Login and registration page functions
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
    return register_user(first_name, last_name, email, password, role)


# Azure API Endpoint for Upload
UPLOAD_API_URL = TRIGGER_URL

# Global variable to track the logged-in user's email
current_user_email = None  # No user is logged in initially

# Function to handle video uploads
def upload_video(file):
    if current_user_email is None:
        return "You must log in before uploading a video."
    
    if file is None:
        return "Please upload a video file."
    
    # Check if the file is a string or a file-like object
    if isinstance(file, str):
        file_path = file
    else:
        file_path = os.path.join("temp_video.mp4")
        with open(file_path, "wb") as f:
            f.write(file.read())

    # Open the video file and send it to the Azure API
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(UPLOAD_API_URL, files=files)

    # Clean up the temporary file if it was created
    if os.path.exists(file_path):
        os.remove(file_path)

    if response.status_code == 200:
        # Get user ID based on email
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (current_user_email,))
        user_data = cursor.fetchone()
        if user_data:
            user_id = user_data[0]

            # Insert the video details into the Videos table
            cursor.execute("""
                INSERT INTO Videos (UserID, FileName, UploadTime)
                VALUES (?, ?, ?)
            """, (user_id, file_path, datetime.now()))
            conn.commit()

        return "Video uploaded successfully! Processing results..."
    else:
        return f"Upload failed: {response.text}"

# Function to fetch uploaded videos with user info
def get_uploaded_videos():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.FirstName, u.LastName, v.FileName, v.UploadTime
        FROM Videos v
        JOIN Users u ON v.UserID = u.UserID
    """)
    videos = cursor.fetchall()
    return videos

# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Welcome to the Athletics Evaluation System 🏃‍♂️🏃‍♀️")

    # Register Tab
    with gr.Tab("Register"):
        gr.Markdown("## Register New User")
        first_name_input = gr.Textbox(label="First Name")
        last_name_input = gr.Textbox(label="Last Name")
        email_input = gr.Textbox(label="Email")
        password_input_reg = gr.Textbox(label="Password", type="password")
        role_input_reg = gr.Radio(["student", "teacher"], label="Role")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Result", interactive=False)

        # Update inputs and function call for registration
        register_btn.click(
            register_page,
            inputs=[first_name_input, last_name_input, email_input, password_input_reg, role_input_reg],
            outputs=register_output
        )

    # Login Tab
    with gr.Tab("Login"):
        gr.Markdown("## Please Log In")
        email_input_log = gr.Textbox(label="Email")
        password_input_log = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)

        login_btn.click(login_page, inputs=[email_input_log, password_input_log], outputs=login_output)

    # Upload Video Tab
    with gr.Tab("Upload Video"):
        gr.Markdown("## Upload your video for evaluation")
        video_input = gr.Video(label="Upload Video")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")

        upload_btn.click(upload_video, inputs=video_input, outputs=upload_output)

    # View Results Tab
    with gr.Tab("View Results"):
        gr.Markdown("## View Results")
        uploaded_videos = get_uploaded_videos()
        for video in uploaded_videos:
            gr.Markdown(f"**{video[0]} {video[1]}** uploaded: {video[2]} at {video[3]}")

# Launch the app
athletics_app.launch()

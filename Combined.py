import gradio as gr
import pyodbc
import hashlib
import requests
import os
from settings.config import CONNECTION_STRING, TRIGGER_URL
import re


# validate email format
def is_valid_student_email(email):
    # name.lastname@student.howest.be
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@student\.howest\.be$', email))

def is_valid_teacher_email(email):
    # name.lastname@howest.be
    return bool(re.match(r'^[a-zA-Z]+(?:\.[a-zA-Z]+)*@howest\.be$', email))

# validate email format based on role
def validate_email(email, role):
    if role == 'student':
        if not is_valid_student_email(email):
            return "Error: Invalid student email format. It should be name.lastname@student.howest.be"
    elif role == 'teacher':
        if not is_valid_teacher_email(email):
            return "Error: Invalid teacher email format. It should be name.lastname@howest.be"
    return None  # No errors, email is valid


# validate password
def validate_password(password):
    # Check if password is at least 8 characters long
    if len(password) < 8:
        return "Error: Password must be at least 8 characters long."
    
    # Check if password contains spaces
    if ' ' in password:
        return "Error: Password should not contain spaces."
    
    return None  # No errors, password is valid


# Connect to Azure SQL Database
# def connect_to_db():
#     conn = pyodbc.connect(CONNECTION_STRING)
#     return conn

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
        return student_view()
    elif role == "teacher":
        return teacher_view()
    else:
        return "Invalid email or password. Please try again."

def register_page(first_name, last_name, email, password, role):
    return register_user(first_name, last_name, email, password, role)

# Azure API Endpoint for Upload
UPLOAD_API_URL = TRIGGER_URL

# Function to handle video uploads
def upload_video(file):
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
        return "Video uploaded successfully! Processing results..."
    else:
        return f"Upload failed: {response.text}"

# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Login, Register & Upload")

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

    # View Results Tab Placeholder
    with gr.Tab("View Results"):
        gr.Markdown("## View Results")
        gr.Markdown("Results functionality is under development. Please check back later!")

# Launch the app
athletics_app.launch()

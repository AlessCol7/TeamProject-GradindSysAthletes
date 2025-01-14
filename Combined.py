import gradio as gr
import pyodbc
import hashlib
import requests
import os
from datetime import datetime

UPLOAD_API_URL = "http://localhost:7071/api/UploadImageTrigger"

def connect_to_db():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                          'Server=tcp:atish.database.windows.net,1433;'
                          'Database=atish-LoginData;'
                          'Uid=atish;Pwd=13sql17_ctai;'
                          'Encrypt=yes;TrustServerCertificate=no;'
                          'Connection Timeout=30;')
    return conn

# Hash password 
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(first_name, last_name, email, username, password, role):
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Check if the username already exists
    cursor.execute("SELECT COUNT(*) FROM Users WHERE Username = ?", (username,))
    if cursor.fetchone()[0] > 0:
        return f"Username '{username}' is already registered. Please log in or choose a different username."
    
    # Hash the password before storing
    hashed_password = hash_password(password)
    
    # Insert the user data into the Users table
    cursor.execute("INSERT INTO Users (FirstName, LastName, Email, Username, Password, Role) VALUES (?, ?, ?, ?, ?, ?)",
                   (first_name, last_name, email, username, hashed_password, role))
    conn.commit()
    conn.close()
    
    return f"User '{username}' has been successfully registered as {role}."

# Function to validate login credentials
def validate_login(username, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Password, Role FROM Users WHERE Username = ?", (username,))
    user_data = cursor.fetchone()

    if user_data:
        stored_password, role = user_data
        if stored_password == hash_password(password):  # Compare hashed passwords
            return role
    return "Invalid username or password"

# Function to display the student view
def student_view():
    return "Welcome, Student! You can now upload videos for evaluation."

# Function to display the teacher view
def teacher_view():
    return "Welcome, Teacher! You can view all students' videos and results."

# Login and registration page functions
def login_page(username, password):
    role = validate_login(username, password)
    if role == "student":
        return student_view()
    elif role == "teacher":
        return teacher_view()
    else:
        return "Invalid username or password. Please try again."

def register_page(first_name, last_name, email, username, password, role):
    return register_user(first_name, last_name, email, username, password, role)

# Function to log video uploads
def log_video_upload(username, video_filename):
    conn = connect_to_db()
    cursor = conn.cursor()
    # Insert upload info into the VideoUploads table
    cursor.execute("INSERT INTO VideoUploads (Username, VideoFilename, UploadTime) VALUES (?, ?, ?)",
                   (username, video_filename, datetime.now()))
    conn.commit()


def upload_video(file, username):
    if file is None:
        return "Please upload a video file."
    
    # Check if the file is a string or a file-like object
    if isinstance(file, str):
        file_path = file  # If it's a file path, use it directly
        video_filename = os.path.basename(file_path)  # Extract the filename
    else:
        # If it's a file-like object, save it temporarily
        file_path = os.path.join("temp_video.mp4")
        with open(file_path, "wb") as f:
            f.write(file.read())
        video_filename = file.name  # Use the name attribute of the file object

    # Open the video file and send it to the Azure API
    with open(file_path, "rb") as f:
        files = {"file": f}
        
        # Post to Azure API
        response = requests.post(UPLOAD_API_URL, files=files)
    
    # Clean up the temporary file if it was created
    if os.path.exists(file_path):
        os.remove(file_path)
    
    if response.status_code == 200:
        # Log the video upload in the database
        log_video_upload(username, video_filename)
        return "Video uploaded successfully! Processing results..."
    else:
        return f"Upload failed: {response.text}"

    
# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Login & Register")
    
    # Register Form
    with gr.Tab("Register"):
        gr.Markdown("## Register New User")
        first_name_input_reg = gr.Textbox(label="First Name")
        last_name_input_reg = gr.Textbox(label="Last Name")
        email_input_reg = gr.Textbox(label="Email")
        username_input_reg = gr.Textbox(label="Username")
        password_input_reg = gr.Textbox(label="Password", type="password")
        role_input_reg = gr.Radio(["student", "teacher"], label="Role")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Result", interactive=False)
        
        register_btn.click(register_page, inputs=[first_name_input_reg, last_name_input_reg, email_input_reg, username_input_reg, password_input_reg, role_input_reg], outputs=register_output)
    
    # Login Form
    with gr.Tab("Login"):
        gr.Markdown("## Please log in")
        
        username_input_log = gr.Textbox(label="Username")
        password_input_log = gr.Textbox(label="Password", type="password")
        
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)
        
        login_btn.click(login_page, inputs=[username_input_log, password_input_log], outputs=login_output)

    # Upload Section
    with gr.Tab("Upload Video"):
        gr.Markdown("## Upload your video for evaluation")
        video_input = gr.Video(label="Upload Video")
        username_input_upload = gr.Textbox(label="Username (for tracking)", interactive=False)
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")
        
        upload_btn.click(upload_video, inputs=[video_input, username_input_upload], outputs=upload_output)
    
    # Results Section Placeholder
    with gr.Tab("View Results"):
        gr.Markdown("## View Results")
        gr.Markdown("Results functionality is under development. Please check back later!")

# Launch the app
athletics_app.launch()

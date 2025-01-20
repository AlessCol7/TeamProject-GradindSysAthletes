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

# def register_page(first_name, last_name, email, password, role):
#     return register_user(first_name, last_name, email, password, role)
def register_page(first_name, last_name, email, password, role):
    if not first_name.strip():
        return "Error: First name is required."
    if not last_name.strip():
        return "Error: Last name is required."
    return register_user(first_name, last_name, email, password, role)


UPLOAD_API_URL = TRIGGER_URL
current_user_email = None

def upload_video(file, sport_branch):
    if current_user_email is None:
        return "You must log in before uploading a video."

    if file is None:
        return "Please upload a video file."

    if sport_branch not in ["Sprint Start", "Sprint Running", "Shot Put", "Relay Receiver", "Long Jump", "Javelin", "High Jump", "Discus Throw", "Hurdling"]:
        return "Invalid sport branch selection."

    if isinstance(file, str):
        file_path = file
    else:
        file_path = os.path.join("temp_video.mp4")
        with open(file_path, "wb") as f:
            f.write(file.read())

    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(UPLOAD_API_URL, files=files)

    if os.path.exists(file_path):
        os.remove(file_path)

    if response.status_code == 200:
        conn = connect_to_db()
        cursor = conn.cursor()

        cursor.execute("SELECT UserID FROM Users WHERE Email = ?", (current_user_email,))
        user_data = cursor.fetchone()
        if user_data:
            user_id = user_data[0]

            cursor.execute("""
                INSERT INTO Videos (UserID, FileName, UploadTime, SportBranch)
                VALUES (?, ?, ?, ?)
            """, (user_id, file_path, datetime.now(), sport_branch))
            conn.commit()

        return f"Video uploaded successfully for {sport_branch}! Processing results..."
    else:
        return f"Upload failed: {response.text}"

# Define get_uploaded_videos function
def get_uploaded_videos():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.FileName, v.SportBranch, v.UploadTime, u.Email
        FROM Videos v
        INNER JOIN Users u ON v.UserID = u.UserID
    """)
    return cursor.fetchall()

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
        gr.Markdown("## View Results")
        uploaded_videos = get_uploaded_videos()
        for video in uploaded_videos:
            gr.Markdown(f"**{video[0]} {video[1]}** uploaded: {video[2]} by {video[3]}")

athletics_app.launch()

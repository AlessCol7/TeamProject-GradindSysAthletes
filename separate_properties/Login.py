import gradio as gr
import pyodbc
import hashlib

# Connect to Azure SQL Database
def connect_to_db():
    conn = pyodbc.connect('Driver={.}')

    return conn

# Hash password (for security purposes, use bcrypt or argon2 in production)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to insert a new user into the database
def register_user(username, password, role):
    # Hash the password before storing
    hashed_password = hash_password(password)
    
    conn = connect_to_db()
    cursor = conn.cursor()
    # Insert user data into the Users table
    cursor.execute("INSERT INTO Users (Username, Password, Role) VALUES (?, ?, ?)",
                   (username, hashed_password, role))
    conn.commit()
    return f"User '{username}' registered successfully as {role}."

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

def register_page(username, password, role):
    return register_user(username, password, role)

# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Login & Register")
    
    # Register Form
    with gr.Tab("Register"):
        gr.Markdown("## Register New User")
        username_input_reg = gr.Textbox(label="Username")
        password_input_reg = gr.Textbox(label="Password", type="password")
        role_input_reg = gr.Radio(["student", "teacher"], label="Role")
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Result", interactive=False)
        
        register_btn.click(register_page, inputs=[username_input_reg, password_input_reg, role_input_reg], outputs=register_output)
    
    # Login Form
    with gr.Tab("Login"):
        gr.Markdown("## Please log in")
        
        username_input_log = gr.Textbox(label="Username")
        password_input_log = gr.Textbox(label="Password", type="password")
        
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)
        
        login_btn.click(login_page, inputs=[username_input_log, password_input_log], outputs=login_output)

# Launch the app
athletics_app.launch()

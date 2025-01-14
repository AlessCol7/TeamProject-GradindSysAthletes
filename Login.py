import gradio as gr
import pyodbc
import hashlib

# Connect to Azure SQL Database
def connect_to_db():
    conn = pyodbc.connect('Server=tcp:atish.database.windows.net,1433;Initial Catalog=atish-LoginData;Persist Security Info=False;User ID=atish;Password=13sql17_ctai;MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;')
    return conn

# Simulate password hashing for demonstration purposes (use a proper hashing mechanism in production)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to validate login credentials from the database
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

# Login page function
def login_page(username, password):
    role = validate_login(username, password)
    if role == "student":
        return student_view()
    elif role == "teacher":
        return teacher_view()
    else:
        return "Invalid username or password. Please try again."

# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Login")
    
    # Login form
    with gr.Tab("Login"):
        gr.Markdown("## Please log in")
        
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)
        
        login_btn.click(login_page, inputs=[username_input, password_input], outputs=login_output)

# Launch the app
athletics_app.launch()

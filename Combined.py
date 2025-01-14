import gradio as gr
import pyodbc
import hashlib
import re

def validate_email(email):
    email = email.strip()  # Strip any leading/trailing whitespace
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zAZ0-9.-]+\.[a-zA-Z]{2,}$'
    print(f"Validating email: '{email}'")  
    if re.match(email_regex, email):
        print(f"Valid email: {email}")  
        return True
    else:
        print(f"Invalid email: {email}")  
        return False

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
    # Strip any extra spaces from input
    email = email.strip()  
    first_name = first_name.strip()  # Optional: strip first name
    last_name = last_name.strip()  # Optional: strip last name
    
    # Debugging: Print values received
    print(f"Received email: '{email}'")  # Log the email
    print(f"Received first name: '{first_name}'")
    print(f"Received last name: '{last_name}'")
    print(f"Received role: '{role}'")
    
    # Validate email format
    if not validate_email(email):
        return "Invalid email format. Please enter a valid email address."
    
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    
    if " " in password or " " in email:
        return "Email or password cannot contain spaces."
    
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Check if the email already exists
    cursor.execute("SELECT COUNT(*) FROM Users WHERE Email = ?", (email,))
    if cursor.fetchone()[0] > 0:
        return f"Email '{email}' is already registered. Please log in or choose a different email."
    
    # Hash the password before storing
    hashed_password = hash_password(password)
    
    # Insert the user data into the Users table
    cursor.execute("INSERT INTO Users (Email, Password, FirstName, LastName, Role) VALUES (?, ?, ?, ?, ?)",
                   (email, hashed_password, first_name, last_name, role))
    conn.commit()
    conn.close()
    
    return f"User with email '{email}' has been successfully registered as {role}."

def login_page(email, password):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT Password, Role FROM Users WHERE Email = ?", (email,))
    user_data = cursor.fetchone()

    if user_data:
        stored_password, role = user_data
        if stored_password == hash_password(password):  # Compare hashed passwords
            return role
    return "Invalid email or password"

# Gradio Interface
with gr.Blocks() as athletics_app:
    gr.Markdown("# Athletics App - Register")
    
    # Register Form
    with gr.Tab("Register"):
        gr.Markdown("## Register New User")
        
        # First name, Last name, Email, and Password inputs
        first_name_input_reg = gr.Textbox(label="First Name")
        last_name_input_reg = gr.Textbox(label="Last Name")
        email_input_reg = gr.Textbox(label="Email")
        password_input_reg = gr.Textbox(label="Password", type="password")
        role_input_reg = gr.Radio(["student", "teacher"], label="Role")
        
        # Register Button and Output
        register_btn = gr.Button("Register")
        register_output = gr.Textbox(label="Registration Result", interactive=False)
        
        # Click event for register button
        register_btn.click(register_user, 
                           inputs=[first_name_input_reg, last_name_input_reg, email_input_reg, password_input_reg, role_input_reg], 
                           outputs=register_output)

    # Login Form
    with gr.Tab("Login"):
        gr.Markdown("## Please log in")
        
        email_input_log = gr.Textbox(label="Email")
        password_input_log = gr.Textbox(label="Password", type="password")
        
        login_btn = gr.Button("Login")
        login_output = gr.Textbox(label="Login Result", interactive=False)
        
        login_btn.click(login_page, inputs=[email_input_log, password_input_log], outputs=login_output)

# Launch the app
athletics_app.launch()


from flask import Flask, render_template, request, redirect, url_for
import pyodbc
import os
from Uploadvideo import upload_video  # Import video upload function
from Login import register_user, validate_login  # Import login functions

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    print("Rendering login page...")
    return render_template('login.html')

# Login route
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    
    # Check if the login is valid
    if validate_login(username, password) == role:
        return redirect(url_for('upload_video'))  # Redirect to upload video page
    else:
        return render_template('login.html', error="Invalid credentials")  # Show error on failure

# Registration route
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']
    
    if register_user(username, password, role):
        return redirect(url_for('login'))  # After registration, redirect to login page
    else:
        return render_template('login.html', error="Registration failed")  # Show error on failure

# Upload video route
@app.route('/upload', methods=['GET', 'POST'])
def upload_video_page():
    if request.method == 'POST':
        video_file = request.files['video']
        video_path = os.path.join('uploads', video_file.filename)
        video_file.save(video_path)
        
        # Call the upload_video function from Uploadvideo.py
        upload_status = upload_video(video_path)  # Handle upload in Uploadvideo.py
        return render_template('upload.html', message=upload_status)  # Show upload status message
    
    return render_template('upload.html')  # Render the upload form for GET request

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# from flask import Flask, request, redirect, jsonify
# from azure.storage.blob import BlobServiceClient
# import pyodbc
# import os

# app = Flask(__name__)

# # Azure Storage and SQL Configuration
# AZURE_STORAGE_CONNECTION_STRING = 'Driver={ODBC Driver 17 for SQL Server};'
# 'Server=tcp:atish.database.windows.net,1433;'
# 'Database=atish-LoginData;'
# 'Uid=atish;Pwd=13sql17_ctai;'
# 'Encrypt=yes;TrustServerCertificate=no;'
# 'Connection Timeout=30;'

# BLOB_CONTAINER_NAME = "videos"
# SQL_SERVER = "atish.database.windows.net"
# SQL_DATABASE = "atish-LoginData"
# SQL_USERNAME = "atish"
# SQL_PASSWORD = "13sql17_ctai"

# # Database connection
# def get_db_connection():
#     conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
#                           f'Server={SQL_SERVER};'
#                           f'Database={SQL_DATABASE};'
#                           f'UID={SQL_USERNAME};'
#                           f'PWD={SQL_PASSWORD};')
#     return conn

# @app.route('/upload', methods=['POST'])
# def upload_video():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400
    
#     # Upload video to Azure Blob Storage
#     blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#     blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=file.filename)
    
#     blob_client.upload_blob(file.read(), overwrite=True)

#     # Save video metadata to Azure SQL
#     video_url = f"https://your_storage_account.blob.core.windows.net/{BLOB_CONTAINER_NAME}/{file.filename}"
#     user_id = request.form.get('user_id')  # Assuming user_id is passed in the form data

#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO videos (video_url, user_id) VALUES (?, ?)", video_url, user_id)
#     conn.commit()
#     conn.close()

#     return jsonify({"message": "File uploaded successfully", "video_url": video_url}), 201

# @app.route('/video/<int:video_id>', methods=['GET'])
# def get_video(video_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT video_url FROM videos WHERE video_id = ?", video_id)
#     row = cursor.fetchone()
#     conn.close()

#     if row:
#         return redirect(row[0])
#     else:
#         return jsonify({"error": "Video not found"}), 404

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Welcome to the Flask App! Visit /upload to upload videos."

# Upload route (for example purposes)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        return "Video uploaded!"
    return "Upload your video here."

if __name__ == '__main__':
    app.run(debug=True)

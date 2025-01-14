import gradio as gr
import requests
import os

# Azure API Endpoint for Upload
UPLOAD_API_URL = "http://localhost:7071/api/UploadImageTrigger"

# Function to handle video uploads
def upload_video(file):
    if file is None:
        return "Please upload a video file."
    
    # Check if the file is a string or a file-like object
    if isinstance(file, str):
        file_path = file  # If it's a file path, use it directly
    else:
        # If it's a file-like object, save it temporarily
        file_path = os.path.join("temp_video.mp4")
        with open(file_path, "wb") as f:
            f.write(file.read())
    
    # Open the video file and send it to the Azure API
    with open(file_path, "rb") as f:
        files = {"file": f}
        
        # Post to Azure API
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
    gr.Markdown("# Athletics App - Automated Athletics Evaluation")
    
    # Upload Section
    with gr.Tab("Upload Video"):
        gr.Markdown("## Upload your video for evaluation")
        video_input = gr.Video(label="Upload Video")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Status")
        
        upload_btn.click(upload_video, inputs=video_input, outputs=upload_output)
    
    # Results Section Placeholder
    with gr.Tab("View Results"):
        gr.Markdown("## View Results")
        gr.Markdown("Results functionality is under development. Please check back later!")
    
# Launch the app
athletics_app.launch()


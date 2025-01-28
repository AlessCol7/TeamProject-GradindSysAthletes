import gradio as gr
from azure.storage.blob import BlobServiceClient
import os

# Define Azure Blob Storage details
SAS_TOKEN = "https://atishstorage.blob.core.windows.net/videos?sp=racw&st=2025-01-26T19:12:49Z&se=2025-01-27T03:12:49Z&spr=https&sv=2022-11-02&sr=c&sig=181sRDcrLmtUp7Mp53zO2oN70C9sTr14QSFI8Nd9hQg%3D?sp=racw&st=2025-01-26T19:12:49Z&se=2025-01-27T03:12:49Z&spr=https&sv=2022-11-02&sr=c&sig=181sRDcrLmtUp7Mp53zO2oN70C9sTr14QSFI8Nd9hQg%3D"  # Replace with your SAS token
STORAGE_ACCOUNT_URL = "https://atishstorage.blob.core.windows.net/"
CONTAINER_NAME = "videos"

# Function to list blobs in the container
def list_videos():
    try:
        # Create BlobServiceClient using the SAS token
        blob_service_client = BlobServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=SAS_TOKEN)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        # List blobs in the container
        blob_urls = []
        for blob in container_client.list_blobs():
            blob_url = f"{STORAGE_ACCOUNT_URL}{CONTAINER_NAME}/{blob.name}?{SAS_TOKEN}"
            blob_urls.append(blob_url)

        return blob_urls
    except Exception as e:
        return f"Error: {str(e)}"

# Function to display videos
def display_videos():
    videos = list_videos()
    if isinstance(videos, list):
        # Return videos in an HTML format
        video_html = ""
        for video_url in videos:
            video_html += f"""
            <div style='margin-bottom: 20px;'>
                <video controls width="400">
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <p>{video_url}</p>
            </div>
            """
        return video_html
    else:
        return f"Error fetching videos: {videos}"

# Define Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Azure Blob Storage Video Viewer")
    with gr.Row():
        view_button = gr.Button("View Uploaded Videos")
    output = gr.HTML()
    view_button.click(display_videos, inputs=[], outputs=output)

# Launch the app
demo.launch()

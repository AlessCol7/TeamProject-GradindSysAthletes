import os
from yt_dlp import YoutubeDL
from bs4 import BeautifulSoup

# Define the folder containing HTML files
folder_path = 'YouTubeDownload/Movies Students'

# Define the subfolder path where videos will be saved
subfolder = 'YouTubeDownload/DownloadedVideos'

# Create the subfolder if it does not exist
if not os.path.exists(subfolder):
    os.makedirs(subfolder)

# Define options for the download
options = {
    'format': 'best',
    'outtmpl': os.path.join(subfolder, '%(title)s.%(ext)s'),
    'force_generic_extractor': True,  # Force generic extractor
}


# Function to extract YouTube links from HTML files
def extract_video_links_from_html(folder_path):
    video_links = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):  # Only consider .html files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Parse the HTML content
                soup = BeautifulSoup(file, 'html.parser')
                # Find the meta tag with the redirect URL
                meta_tag = soup.find('meta', {'http-equiv': 'Refresh'})
                if meta_tag:
                    # Extract the URL from the content attribute
                    content = meta_tag.get('content')
                    if content:
                        # The URL is after "0; url="
                        video_url = content.split('url=')[-1]
                        video_links.append(video_url)
    return video_links

# Extract video links from the folder
video_links = extract_video_links_from_html(folder_path)

# Download each video, handle errors gracefully
if video_links:
    with YoutubeDL(options) as ydl:
        for link in video_links:
            try:
                print(f"Downloading: {link}")
                ydl.download([link])
            except Exception as e:
                print(f"Error downloading {link}: {e}")
    print("All downloads complete!")
else:
    print("No video links found in the folder.")


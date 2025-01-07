import cv2
import os
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Directory paths
video_dir = "Athletes/YouTubeDownload/DownloadedVideos"
output_frame_dir = "Athletes/Segmentation/SavedFrames"

# Function to detect scenes (content changes) in a video
def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    
    # List of scene start/end frame numbers
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(start.get_frames(), end.get_frames()) for start, end in scene_list]

# Function to extract frames for a specific range
def extract_frames(video_path, output_dir, start_frame, end_frame, frame_interval=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret or frame_count > end_frame:
            break

        if start_frame <= frame_count <= end_frame and frame_count % frame_interval == 0:
            filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    video_capture.release()

# Process all videos in the directory
for video_filename in os.listdir(video_dir):
    if not video_filename.endswith(".mp4"):
        continue

    video_path = os.path.join(video_dir, video_filename)
    print(f"Processing video: {video_filename}")
    
    # Detect scenes in the video
    scenes = detect_scenes(video_path, threshold=30.0)
    print(f"Detected {len(scenes)} scenes in {video_filename}")

    # Extract frames for each scene
    for i, (start_frame, end_frame) in enumerate(scenes):
        technique_output_dir = os.path.join(output_frame_dir, os.path.splitext(video_filename)[0], f"Scene_{i + 1}")
        print(f"Extracting frames for Scene {i + 1} ({start_frame}-{end_frame})")
        extract_frames(video_path, technique_output_dir, start_frame, end_frame)

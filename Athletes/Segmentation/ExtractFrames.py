import cv2
import os
import matplotlib.pyplot as plt

video_dir = "Athletes/YouTubeDownload/DownloadedVideos"
output_frame_dir = "Athletes/Segmentation/SavedFrames"  # Path to save the extracted frames

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_dir, frame_interval=20):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Save frames at specific intervals (e.g., every 'frame_interval' frames)
        if frame_count % frame_interval == 0:
            # Save the frame
            filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, frame)
            extracted_frames.append((frame, filename))
        
        frame_count += 1
    
    video_capture.release()
    return extracted_frames

# Function to visualize a few frames
def show_frames(frames, filenames):
    for i, (frame, filename) in enumerate(zip(frames, filenames)):
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title(filename)
        plt.show()

# Function to segment the frames into exercises (same as before)
def segment_frames(frames, filenames, segments):
    segmented_data = {}
    
    for exercise, (start, end) in segments.items():
        segmented_frames = frames[start:end]
        segmented_filenames = filenames[start:end]
        segmented_data[exercise] = {
            "frames": segmented_frames,
            "filenames": segmented_filenames
        }
    
    return segmented_data

# Define your exercise segments here (same as before)
exercise_segments = {
    '1-Sprint Start': (300, 900),   
    '2-Sprint Running': (920, 1220), 
    '3-Shot Put': (1240, 1900),  
    '4-High Jump': (1920, 2680),  
    '5-Hurdling':(2700, 3000),
    '6-Long Jump': (3200, 4000), 
    '7-Discus Throw': (4200,6000),  
    '8-Javelin': (6020,7000),  
    '9-Relay receiver': (7020,7500)
}

# Extract frames from each video in the video directory
for video_filename in os.listdir(video_dir):
    if video_filename.endswith('.mp4'):
        video_path = os.path.join(video_dir, video_filename)
        print(f"Processing video: {video_filename}")
        
        # Define an output directory for each video
        video_output_dir = os.path.join(output_frame_dir, video_filename.split('.')[0])
        
        # Extract frames from the video
        extracted_frames = extract_frames_from_video(video_path, video_output_dir)
        
        # Separate frames and filenames
        frames = [frame for frame, _ in extracted_frames]
        filenames = [filename for _, filename in extracted_frames]
        
        # Segment the frames into exercise segments
        segmented_data = segment_frames(frames, filenames, exercise_segments)
        
        # Show or save segmented exercises (e.g., for Sprint Start)
        for exercise, data in segmented_data.items():
            print(f"Displaying frames for {exercise} from video {video_filename}")
            show_frames(data['frames'], data['filenames'])
            
        # Optionally, you can save the segmented frames into specific folders for each exercise
        for exercise, data in segmented_data.items():
            exercise_dir = os.path.join(output_frame_dir, video_filename.split('.')[0], exercise)
            if not os.path.exists(exercise_dir):
                os.makedirs(exercise_dir)
                
            for frame, filename in zip(data['frames'], data['filenames']):
                cv2.imwrite(os.path.join(exercise_dir, filename), frame)


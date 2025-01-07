import os
import cv2
import csv

# Define the segments and corresponding exercises
exercise_segments = {
    'Sprint Start': (300, 900),
    'Sprint Running': (920, 1220),
    'Shot Put': (1240, 1900),
    'High Jump': (1920, 2680),
    'Hurdling': (2700, 3000),
    'Long Jump': (3050, 3800),
    'Discus Throw': (3850, 4500),
    'Javelin': (4600, 5300),
    'Relay Receiver': (5400, 6000)
}

# Function to label the frames based on time range
def label_frames(video_path, exercise_segments):
    # Open the video
    video = cv2.VideoCapture(video_path)
    frame_rate = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    labels = []
    
    for frame_number in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break
        
        time_in_seconds = frame_number / frame_rate
        
        for exercise, (start_time, end_time) in exercise_segments.items():
            if start_time <= time_in_seconds <= end_time:
                labels.append((frame_number, exercise))
                break

    return labels

# Process the video and generate the labeled dataset
def create_labeled_data(video_path, output_csv):
    labels = label_frames(video_path, exercise_segments)

    # Write labels to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_filename', 'exercise_label'])
        for frame_number, label in labels:
            frame_filename = f"frame_{frame_number:04d}.jpg"
            writer.writerow([frame_filename, label])

# Example usage
video_path = 'Athletes/Segmentation/SavedFrames'
output_csv = 'labeled_frames.csv'
create_labeled_data(video_path, output_csv)

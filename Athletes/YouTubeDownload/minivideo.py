import os
import cv2
import pytesseract
import moviepy.editor as mp

# Set ImageMagick for moviepy compatibility
os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"

# Function to detect text on the white paper
def detect_text_in_frame(frame):
    # Use Tesseract OCR to extract text from the frame
    text = pytesseract.image_to_string(frame)
    return text.strip()

# Function to segment the video based on exercise criteria (white paper detection)
def segment_video(input_video_path, output_folder):
    # Open the video
    video_capture = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    exercise_number = 1
    last_paper_time = None
    segments = []

    # Iterate through each frame of the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        current_time = frame_count / frame_rate

        # Convert to grayscale to improve text detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding to isolate bright white paper
        _, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)

        # If a white paper with text is detected
        text = detect_text_in_frame(frame)

        if text:  # If there's text detected (indicating a paper frame)
            print(f"Detected criteria: {text}")
            
            # If last paper was found, cut the video from last paper to this paper
            if last_paper_time is not None:
                start_time = last_paper_time
                end_time = current_time
                segment_clip = mp.VideoFileClip(input_video_path).subclip(start_time, end_time)

                # Save the segment video
                segment_clip.write_videofile(f"{output_folder}/exercise_{exercise_number}.mp4", codec="libx264")

                exercise_number += 1  # Increment the exercise number

            # Update last_paper_time with the current time
            last_paper_time = current_time

    # Release the video capture
    video_capture.release()

# Example usage
segment_video('/Users/alessiacolumban/Desktop/Athletes/YouTubeDownload/DownloadedVideos/atletiek.mp4', '/Users/alessiacolumban/Desktop/Athletes/YouTubeDownload/minivideos')

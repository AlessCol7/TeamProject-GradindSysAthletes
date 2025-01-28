
import cv2
from ultralytics import YOLO
from yt_dlp import YoutubeDL
import os

# --------------------- Video Download ---------------------
def download_video(video_link, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    options = {
        'format': 'best',
        'outtmpl': os.path.join(download_folder, '%(title)s.%(ext)s'),
    }

    with YoutubeDL(options) as ydl:
        info_dict = ydl.extract_info(video_link, download=True)
        return ydl.prepare_filename(info_dict)

# --------------------- Video Segmentation ---------------------
def segment_video(video_path, segment_folder, yolo_model, min_detection_frames=5, detection_buffer=10):
    if not os.path.exists(segment_folder):
        os.makedirs(segment_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {video_path}")
        return []

    frame_count = 0
    detection_count = 0
    no_detection_buffer = 0
    is_recording = False
    video_writer = None
    segments = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = yolo_model(frame)

        # Closest box variables
        closest_bbox = None
        closest_distance = float("inf")  # Initialize to a very large value
        max_bbox_area = 0

        for result in results:
            if result.boxes:
                for bbox, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    if score > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)

                        # Calculate the center of the bounding box and the frame center
                        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)
                        distance = ((bbox_center[0] - frame_center[0]) ** 2 + (bbox_center[1] - frame_center[1]) ** 2) ** 0.5

                        # Update closest box logic
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_bbox = bbox
                            max_bbox_area = area

        # Determine if detection is valid
        has_detection = closest_bbox is not None

        if has_detection:
            detection_count += 1
            no_detection_buffer = 0
        else:
            no_detection_buffer += 1

        # Start recording if enough consecutive detections occur
        if detection_count >= min_detection_frames and not is_recording:
            is_recording = True
            output_path = os.path.join(segment_folder, f"segment_{frame_count:06d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
            print(f"Started recording: {output_path}")
            segments.append(output_path)

        # Stop recording if no detection for too many consecutive frames
        if no_detection_buffer > detection_buffer and is_recording:
            is_recording = False
            if video_writer:
                video_writer.release()
            print(f"Stopped recording at frame {frame_count}.")
            detection_count = 0

        # Write frames if recording
        if is_recording and video_writer:
            video_writer.write(frame)

    # Release resources
    cap.release()
    if video_writer:
        video_writer.release()

    print("Segmentation complete.")
    return segments

# --------------------- Main Workflow ---------------------
# --------------------- Main Workflow ---------------------
if __name__ == "__main__":
    # Prompt the user for input
    input_type = input("Enter '1' to provide a YouTube URL or '2' for a local video file: ").strip()

    # Load YOLO model
    yolo_model = YOLO("yolov8s-pose.pt")

    if input_type == "1":
        # Prompt for YouTube URL
        video_url = input("Please enter the YouTube video URL: ").strip()

        # Video download
        download_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/DownloadVideoTest"
        video_path = download_video(video_url, download_folder)

    elif input_type == "2":
        # Prompt for local file path
        video_path = input("Please enter the path to the local video file: ").strip()

        # Check if the file exists
        if not os.path.exists(video_path):
            print(f"Error: The file '{video_path}' does not exist.")
            exit(1)

    else:
        print("Invalid input. Please enter '1' or '2'.")
        exit(1)

    # Segmentation
    segment_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/SegmentedVideosOriginal"
    segments = segment_video(video_path, segment_folder, yolo_model)

    print("Segmentation complete. Segments saved in:", segment_folder)


    # # Load YOLO model
    # yolo_model = YOLO("yolov8s-pose.pt")

    # # Video download
    # download_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/DownloadVideoTest"
    # video_path = download_video(video_url, download_folder)

    # # Segmentation
    # segment_folder = "/Users/alessiacolumban/Desktop/TeamProject-GradindSysAthletes/Athletes/Testing/SegementedVideos"
    # segments = segment_video(video_path, segment_folder, yolo_model)
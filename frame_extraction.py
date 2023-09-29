import cv2
import os

#Input and Output directory
video_path = "/Users/jeremydasa/Desktop/MEF_E_Master/Raw_Video_Data/June 19, 2023 (Jeremy) .MOV"
output_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/Extracted_Frames/Jeremy-Frames"

# Function to extract frames (every 10 seconds)
def extract_frames(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_interval = int(fps) * 10  # Extract a frame every 10 seconds

    # Start frame extraction
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_output_path = os.path.join(output_dir, f"frame-{frame_count}.jpg")
            cv2.imwrite(frame_output_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Frames extracted: {frame_count}")
    print(f"Video duration: {video_duration:.2f} seconds")

# Call the function to extract frames
extract_frames(video_path, output_dir)

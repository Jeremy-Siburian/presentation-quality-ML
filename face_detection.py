import cv2
import os
from mtcnn import MTCNN

# Input and Output directory
frames_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/Extracted_Frames/Jeremy-Frames"
output_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/Frames_Annotated/Jeremy-Frames_Annotated"

# Function to detect and label faces in each frame
def face_detection(frames_dir, output_dir):
    detector = MTCNN()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = os.listdir(frames_dir)
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Draw bounding boxes and labels around each face
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            label = f"Face {i+1}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save the frame with detected faces
        output_path = os.path.join(output_dir, f"{frame_file}_annotated")
        cv2.imwrite(output_path, frame)

# Call the function to run
face_detection(frames_dir, output_dir)

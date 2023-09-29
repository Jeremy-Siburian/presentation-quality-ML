import cv2
import os
from mtcnn import MTCNN

frames_dir = "/Users/jeremydasa/Desktop/MEF_E_Master/Frames_Annotated/Jeremy-Frames_Annotated"

# Function to manually label faces and sort depending on label
def face_labeling(frames_dir):
    # Create the output directories for Attentive and Distractive labels for sorting
    output_attentive_dir = os.path.join(frames_dir, "Attentive")
    output_distractive_dir = os.path.join(frames_dir, "Distractive")

    if not os.path.exists(output_attentive_dir):
        os.makedirs(output_attentive_dir)

    if not os.path.exists(output_distractive_dir):
        os.makedirs(output_distractive_dir)

    # Load the MTCNN face detection model
    detector = MTCNN()

    frame_files = sorted(os.listdir(frames_dir))
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Draw bounding boxes around each face
        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            label = f"Face {i+1}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame with the detected face and wait for user input
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)

            # Get user input for the face label (0 for Distractive, 1 for Attentive)
            user_input = input(f"Input label for {label} (0 for Distractive, 1 for Attentive): ")
            user_input = int(user_input.strip())

            # Save the face ROI to the appropriate output directory
            face_roi = frame[y:y+height, x:x+width]
            if user_input == 0:
                output_path = os.path.join(output_distractive_dir, f"Jeremy_{frame_file}_face{i+1}_Distractive.jpg")
            else:
                output_path = os.path.join(output_attentive_dir, f"Jeremy_{frame_file}_face{i+1}_Attentive.jpg")

            cv2.imwrite(output_path, face_roi)

        cv2.destroyAllWindows()

# Run the function to label and sort faces
face_labeling(frames_dir)


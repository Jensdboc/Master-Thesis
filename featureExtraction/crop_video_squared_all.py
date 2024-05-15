import os

# Install libraries
os.system("pip install opencv-python")

import cv2
import matplotlib.pyplot as plt
from PIL import Image

def crop_and_save_video(inputpath: str, outputpath: str) -> None:
    """
    Crop the input video around the face of the participant, but keep the square detecting in the first frame given a certain offset.
    """
    # Initialise video variables
    video_capture = cv2.VideoCapture(inputpath)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    start_width = width//4
    start_height = 0
    n_width = width//2
    n_height = height*3//4
    size = 224
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(outputpath, fourcc, fps, (size, size))

    count = 0
    offset = 40
    faces_found = False
    first = True
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        cropped_frame_to_detect = frame[start_height:start_height + n_height, start_width:start_width + n_width]

        if not faces_found:
            # Detect face with opencv
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(cropped_frame_to_detect, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
            if len(faces) > 0:
                # Make sure the following frames do not differ to much
                # If they differ, take the previous frame. The face should be able to move so fast
                # So it must have been misrecognized.
                if first:
                    x, y, w, h = faces[0]
                else:
                    # Sometimes there are multiple faces recognized, pick the correct one
                    for face in faces:
                        x_n, y_n, w_n, h_n = face
                        if abs(x_n - x) < 10 and abs(y_n - y) < 10:
                            x = x_n
                            y = y_n
                            w = w_n
                            h = h_n
                            break
    
                # Making sure we are working with a square
                a = (w + h) // 2
                min_x = x
                # max_x = x + w
                max_x = x + a
                min_y = y
                # max_y = y + h
                max_y = y + a
                faces_found = True
                first = False
            else:
                print("No face found, taking coord of last face")

        # True once a face is found, ignore all the frames before that
        if faces_found:
            # Crop the frame
            cropped_frame = cropped_frame_to_detect[min_y - offset:max_y + offset, min_x - offset:max_x + offset]
            resized_frame = cv2.resize(cropped_frame, (size, size), interpolation=cv2.INTER_NEAREST)

            # Write the cropped frame to the output video
            output_video.write(resized_frame)
        else:
            print(f"No face in this frame, not appending them.")

    # Release the video capture and video writer objects
    video_capture.release()
    output_video.release()
    
    print("Cropped video saved successfully at:", outputpath)

root = "/project_ghent/Master-Thesis/no_robot_audio/labelled"
subpath = "SQUARED"

for dir in os.listdir(root):
    if dir == "NO" or dir == "YES":
        for video in os.listdir(f"{root}/{dir}"):
            inputpath = f"{root}/{dir}/{video}"
            outputpath = f"{root}/{subpath}/{dir}/{video[:-4]}_squared.mp4"
            if not os.path.exists(outputpath):
                print(inputpath)
                crop_and_save_video(inputpath, outputpath)


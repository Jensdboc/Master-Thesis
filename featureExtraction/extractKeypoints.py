import os

# Install libraries
os.system("pip install opencv-python mediapipe")

import cv2
import csv
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

import mediapipe as mp
import csv
import cv2
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
LANDMARK_HEADER = [[f"{j}{i}" for i in range(468) for j in ["x", "y", "z"]]]

import mediapipe as mp
import csv
import cv2
import os

import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

LANDMARK_HEADER = [[f"{j}{i}" for i in range(468) for j in ["x", "y", "z"]]]

def detect_and_write_facial_keypoints(video_path: str, outputpath: str):
    """
    Detect and write the keypoints from a video frame per frame to a csv file.
    The header will be of the type: x0, y0 ,z0, x1, y1, z1, ...
    """
    v_cap = cv2.VideoCapture(video_path)
    video_length = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_landmarks = LANDMARK_HEADER.copy()

    # assert video_length == 120, f"{video_length} instead of 120"
    
    for _ in range(video_length):
        success, frame = v_cap.read()
        
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame_landmarks = []
                for landmark in face_landmarks.landmark:
                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                all_landmarks.append(frame_landmarks)
        else:
            print(f"Skipping {video_path}, no landmarks detected in a frame.")
            return
                
    # assert len(all_landmarks) == 121, len(all_landmarks)
    
    with open(outputpath, 'w', newline='') as file:
        writer = csv.writer(file)
        for landmarks in all_landmarks:
            writer.writerow(landmarks)
    
    v_cap.release()

root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_split"
output_root = "/project_ghent/Master-Thesis/featureExtraction/output_keypoints_split"
# root = "/project_ghent/Master-Thesis/gif/data"
# output_root = "/project_ghent/Master-Thesis/gif/keypoints"

for dir in os.listdir(root):
    # if dir == "NO":
    if dir == "NO" or dir == "YES":
        for video in os.listdir(f"{root}/{dir}"):
            inputpath = f"{root}/{dir}/{video}"
            if ".ipynb_checkpoints" not in inputpath and ".nfs" not in inputpath:
                outputpath = f"{output_root}/{dir}/{video[:-4]}.csv"
                if not os.path.exists(outputpath):
                    print(inputpath)
                    detect_and_write_facial_keypoints(inputpath, outputpath)
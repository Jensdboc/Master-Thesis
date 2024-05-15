import os

# Install libraries
os.system("pip install opencv-python mediapipe")

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List
from PIL import Image
from mediapipe.tasks import python as mp_python
import subprocess

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dir", type=str)

args = parser.parse_args()

MP_TASK_FILE = "/project_ghent/Master-Thesis/featureExtraction/face_landmarker_with_blendshapes.task"

class FaceMeshDetector:
    """
    Class for extracting keypoints and blendshapes from a video given the videopath.
    """
    def __init__(self) -> None:
        """
        Initialize variable and the mediapipe model.
        """
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_facial_transformation_matrixes=True,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                )
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)

        self.landmarks = []
        self.video_length = 0

    def mp_callback(self, mp_result: mp.tasks.vision.FaceLandmarkerResult) -> None:
        """
        Append landmarks in case there are landmarks detected, else 0.
        """
        if len(mp_result.face_landmarks) >= 1:
            self.landmarks.append(mp_result.face_landmarks[0])
        else:
            # It could be that the first frame does not have landmarks
            # So we cannot simply append the last detected landmarks
            self.landmarks.append(0)
            
    def update_from_video_path(self, video_path: str, to_crop: bool=False) -> None:
        """
        Loop over the given video and detect the given landmarks by calling mp_callback.
        """
        v_cap = cv2.VideoCapture(video_path)
        self.video_length = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in range(self.video_length):
            success, frame = v_cap.read()
            
            if frame is not None:
                frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = self.model.detect(frame_mp)
                self.mp_callback(results)

    def get_results(self) -> Tuple[List[List[mp.tasks.components.containers.NormalizedLandmark]], int]:
        """
        Return the acquired landmarks and video_length.
        """
        return self.landmarks, self.video_length

def extract_mediapipe_info(video_path: str, to_crop: bool=False) -> Tuple[List[List[mp.tasks.components.containers.NormalizedLandmark]], int]:
    """
    Extract landmarks based on the given video_path.
    """
    detector = FaceMeshDetector()
    detector.update_from_video_path(video_path, to_crop)
    return detector.get_results()

def calc_square_distance(tup1, tup2) -> np.float64:
    """
    Calculate the square distance, given two 2D coordinates.
    """
    return np.sqrt(abs((tup1[0] - tup2[0])**2 - (tup1[1] - tup2[1])**2), dtype=np.float64) 

def main_difference_keypoint(keypoints: List[List[mp.tasks.components.containers.NormalizedLandmark]]) -> List[Tuple[int, int]]:
    """
    Calculate the most interesting start and end frames of different fragments based on the difference in keypoints.
    """
    amount_of_fragments = 3
    length_of_fragment = 120
    window_size = 45
    differences = []
    prev_coords = []
    
    # Calculate the difference of the previous frame for every landmark and keep the sum
    for landmarks in keypoints:
        sum_diff = 0
        # Check case if no landmarks were detected in a frame
        if landmarks != 0:
            # If first frame
            if prev_coords == []:
                for landmark in landmarks:
                    prev_coords.append((landmark.x, landmark.y))      
            else:
                coords = []
                for landmark in landmarks:
                    coords.append((landmark.x, landmark.y))
                    sum_diff += calc_square_distance(prev_coords[len(coords) - 1], (landmark.x, landmark.y))
                prev_coords = coords
        differences.append(sum_diff)

    differences = np.convolve(differences, np.ones(window_size)/window_size, mode='same')[window_size//2:-window_size//2]
        
    count = -1
    cut_frames = []
    sorted_max = np.argsort(differences)

    # Append cut_frames until the amount of required fragments but avoid index error
    # The restrictions make sure that only frames that are at least 60 frames later or earlier get chosen
    # This increases the chance to locate multiple interesting moments, 
    # otherwise the same fragments will be chosen for {amount_of_fragments} times
    while len(cut_frames) != amount_of_fragments and abs(count) != len(sorted_max):
        maxi = sorted_max[count]
        if all(maxi > cut + 60 for cut in cut_frames) or all(maxi < cut - 60 for cut in cut_frames):
            cut_frames.append(maxi)
        count -= 1

    # In case the fragments couldn't be chosen with the restrictions, add the most likely fragment multiple times
    # while len(cut_frames) != amount_of_fragments:
    #     cut_frames.append(cut_frames[0])
    # cut_frames = sorted(cut_frames)

    # Calculate start and end for cutting different fragments
    cut_tuple_frames = [(cut - length_of_fragment//2, cut + length_of_fragment//2 - 1) for cut in cut_frames] # -1 for getting a total of 360 frames
    return cut_tuple_frames
        
def save_video(inputpath: str, outputpath: str, frames_to_keep: List[Tuple[int, int]]) -> None:
    """
    Cut the input video based on the given frames and write to the outputpath.
    """
    # Initialize video variables
    temp_path = "/project_ghent/Master-Thesis/featureExtraction/temp.mp4"
    
    video_capture = cv2.VideoCapture(inputpath)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    amount_of_frames = 0
    for start, end in frames_to_keep:
        # Shift start and end such that the amount of frames is always the same
        if end >= v_len:
            start -= end - (v_len - 1)
            end = v_len - 1
        elif start < 0:
            end -= start
            start = 0

        video_capture = cv2.VideoCapture(inputpath)
        count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if count >= start and count <= end:
                output_video.write(frame)
                amount_of_frames += 1
            elif count > end:
                break
            count += 1

    # Check whether the video is the right predefined length of 360 in this case
    # This is when amount_of_fragments = 3 and length_of_fragment = 120
    # assert amount_of_frames == 360, f"Amount of frames is {amount_of_frames}"
    
    video_capture.release()
    output_video.release()

    # Convert to playable format in Jupyter notebooks
    command = [
            "ffmpeg",
            "-y",
            "-i", temp_path,
            "-c:v", "libx264",
            "-crf", "23",
            "-c:a", "aac",
            "-strict", "experimental",
            outputpath
        ]

    # Make sure the surpress the output to avoid too much information in the terminal
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("Video saved successfully at:", outputpath)

# Crop square cropped
# root = "/project_ghent/Master-Thesis/no_robot_audio/labelled/SQUARED"
# output_root = "/project_ghent/Master-Thesis/featureExtraction/output_videos"

# Crop face cropped smooth
root = "/project_ghent/Master-Thesis/no_robot_audio/labelled/CROPPED"
output_root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_face_cropped_smooth"

for dir in os.listdir(root):
    if dir == args.dir:
        for video in os.listdir(f"{root}/{dir}"):
            inputpath = f"{root}/{dir}/{video}"
            outputpath = f"{output_root}/{dir}/{video[:-4]}_extracted.mp4"
            if not os.path.exists(outputpath):
                print(inputpath)
                keypoints, v_len = extract_mediapipe_info(inputpath)
                # Check if enough keypoints are present and if video did not get corrupted:
                if len(keypoints) >= 360:
                    frames_to_keep = main_difference_keypoint(keypoints)
                    save_video(inputpath, outputpath, frames_to_keep)
                
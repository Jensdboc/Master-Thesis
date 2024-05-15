import os

# Install libraries
os.system("pip install opencv-python mediapipe")

import csv
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import subprocess
import pickle

from typing import List
from IPython.display import display, Image, Video
from PIL import Image as PIL_Image
from mediapipe.tasks import python as mp_python

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dir", type=str)

args = parser.parse_args()

MP_TASK_FILE = "/project_ghent/Master-Thesis/featureExtraction/face_landmarker_with_blendshapes.task"
BLENDSHAPE_HEADER = [f"blendshape{i}" for i in range(52)]

class FaceMeshDetector:
    """
    Class for extracting keypoints and blendshapes from a video given the videopath
    """
    def __init__(self) -> None:
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_faces=1,
                )
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)

        self.blendshapes = []
        self.latest_time_ms = 0
        self.video_length = 0

    def mp_callback(self, mp_result, output_image, timestamp_ms: int) -> None:
        if len(mp_result.face_blendshapes) >= 1:
            self.blendshapes.append([b.score for b in mp_result.face_blendshapes[0]])
            
    def update_from_video_path(self, video_path: str, to_crop: bool=False):
        v_cap = cv2.VideoCapture(video_path)
        self.video_length = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(0, self.video_length):
            success, frame = v_cap.read()
            
            if frame is not None:
                frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                results = self.model.detect(frame_mp)
                self.mp_callback(results, None, 0)

    def get_results(self):
        return self.blendshapes, self.video_length

def extract_mediapipe_info(video_path: str, to_crop: bool=False):
    detector = FaceMeshDetector()
    detector.update_from_video_path(video_path, to_crop)
    return detector.get_results()


def detect_and_write_facial_keypoints(video_path: str, outputpath: str):
    """
    Detect and write the keypoints from a video frame per frame to a csv file.
    The header will be of the type: x0, y0 ,z0, x1, y1, z1, ...
    """

    blendshapes, vlen = extract_mediapipe_info(video_path)

    if len(blendshapes) == vlen:
        with open(outputpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(BLENDNAMES)
            for blendshape in blendshapes:
                writer.writerow(blendshape)

BLENDNAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
STRONG_BLENDNAMES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight','eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPucker', 'mouthRight', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight']

# root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_split"
# output_root = "/project_ghent/Master-Thesis/featureExtraction/output_blendshape_split"
# root = "/project_ghent/Master-Thesis/gif/data"
# output_root = "/project_ghent/Master-Thesis/gif/keypoints"
# root = "/project_ghent/Master-Thesis/featureExtraction/output_videos"
# output_root = "/project_ghent/Master-Thesis/featureExtraction/output_blendshape"
root = "/project_ghent/Master-Thesis/featureExtraction/output_video_after_robot_utterance"
output_root = "/project_ghent/Master-Thesis/featureExtraction/output_blendshape_after_robot_utterance"

for dir in os.listdir(root):
    if dir == args.dir:
        for video in os.listdir(f"{root}/{dir}"):
            inputpath = f"{root}/{dir}/{video}"
            if ".ipynb_checkpoints" not in inputpath and ".nfs" not in inputpath:
                outputpath = f"{output_root}/{dir}/{video[:-4]}.csv"
                if not os.path.exists(outputpath):
                    print(inputpath)
                    detect_and_write_facial_keypoints(inputpath, outputpath)
                
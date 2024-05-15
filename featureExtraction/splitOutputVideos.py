import os

# Install libraries
os.system("pip install opencv-python")

import cv2
import subprocess

def cut_video_in_splits(inputpath: str, outputpath: str, amount_of_frames_to_keep: int) -> None:
    """
    Cut the input video into outputvideos.
    """
    video_capture = cv2.VideoCapture(inputpath)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    v_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(v_len // amount_of_frames_to_keep):
        temp_path = f"/project_ghent/Master-Thesis/featureExtraction/temp_{i}.mp4"
        output_video = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        for _ in range(amount_of_frames_to_keep):
            ret, frame = video_capture.read()
            if not ret:
                break
                
            output_video.write(frame)

        output_video.release()

        outputpath_splitvideo = f"{outputpath[:-4]}_{i}.mp4"
        # Convert to playable format in Jupyter notebooks
        command = [
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                "-crf", "23",
                "-c:a", "aac",
                "-strict", "experimental",
                outputpath_splitvideo
            ]
    
        # Make sure the surpress the output to avoid too much information in the terminal
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Video saved successfully at:", outputpath_splitvideo)
        
    video_capture.release()

# root = "/project_ghent/Master-Thesis/featureExtraction/output_videos/"
# output_root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_split/"

root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_face_cropped"
output_root = "/project_ghent/Master-Thesis/featureExtraction/output_videos_face_cropped_split"


for dir in os.listdir(root):
    # if dir == "NO":
    if dir == "YES":
        for video in os.listdir(f"{root}/{dir}"):
            inputpath = f"{root}/{dir}/{video}"
            outputpath = f"{output_root}/{dir}/{video}"
            if not os.path.exists(f"{outputpath[:-4]}_0.mp4") or not os.path.exists(f"{outputpath[:-4]}_1.mp4") or not os.path.exists(f"{outputpath[:-4]}_2.mp4"):
                print(f"{inputpath} -> {outputpath}")
                cut_video_in_splits(inputpath, outputpath, 120)
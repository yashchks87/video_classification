import glob
import os
import cv2
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

class StoreVideoFrames():
    def __init__(self, dir_path, num_of_frames, save_location, img_size = (256, 256)):
        self.dir_path = dir_path
        self.num_of_frames = num_of_frames
        self.save_location = save_location
        self.videos = glob.glob(self.dir_path + '/*.mp4')
        self.img_size = img_size

    def extract_frames(self, video_path):
        video_name = video_path.split('/')[-1].split('.')[0]
        target_path = self.save_location + video_name
        os.makedirs(target_path, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        x1, x2 = 20, 100
        frame_selector = np.arange(x1, x2, (x2-x1)/self.num_of_frames)
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if counter in frame_selector:
                    frame = cv2.resize(frame, (self.img_size[0], self.img_size[0]))
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(target_path + f'/frame_{counter}.jpg', frame)
                    # frame.save(target_path + f'frame_{counter}.jpg')
                counter += 1
            else:
                break
    
    def process_videos(self, is_mp = True, num_of_cores=6):
        if is_mp:
            with mp.Pool(num_of_cores) as p:
                p.map(self.extract_frames, self.videos)
        else:
            for video in tqdm(self.videos):
                self.extract_frames(video)
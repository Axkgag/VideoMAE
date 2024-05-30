from vecpredictor import Vecpredictor
import argparse
import os
import cv2
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
import numpy as np

from StateMachine import StateMachine as SM1
from StateMachine2 import StateMachine as SM2

parser=argparse.ArgumentParser()
parser.add_argument('-t', 
                    '--test_dir', 
                    default='../../data/action/task4/video/0.mp4', 
                    help='data used to build gallery'
                )

parser.add_argument('-m',
                    '--model_dir_path',
                    default='./VEC_MODEL/task1/videomae_ssv2',
                    help='model_dir'
                )

parser.add_argument('-s',
                    '--save_path',
                    default='./TODO/results/videos/test.mp4',
                    help='model_dir'
                )


def get_video_list(video_file):
    video_lists = []
    if video_file is None or not os.path.exists(video_file):
        raise Exception("not found any img file in {}".format(video_file))
    
    if os.path.isfile(video_file):
        return [video_file]

    # img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    video_end = ['mp4']
    if os.path.isfile(video_file) and video_file.split('.')[-1] in video_end:
        video_lists.append(video_file)
    elif os.path.isdir(video_file):
        for single_file in os.listdir(video_file):
            if single_file.split('.')[-1] in video_end:
                video_lists.append(os.path.join(video_file, single_file))
    if len(video_lists) == 0:
        raise Exception("not found any img file in {}".format(video_file))
    video_lists = sorted(video_lists)
    return video_lists

class MediaReader():
    def __init__(self, source, window_size=32, window_stride=16) -> None:
        self.cap = cv2.VideoCapture(source)

        self.stack = deque(maxlen=window_size)
        self.win_size = window_size
        self.win_stride = window_stride
        self.end = False

        self.frame_nums = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 2560
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 1440

    def read(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.stack.append(rgb_frame)
        else:
            self.end = True
        return ret, frame

    def pop(self):
        for _ in range(self.win_stride):
            if self.stack:
                self.stack.popleft()
 
    def release(self, ):
        self.cap.release()


if __name__=="__main__":
    args=parser.parse_args()

    vecpred = Vecpredictor()
    StateMachine = SM1(["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"])

    T = 0

    if vecpred.loadModel(args.model_dir_path):
        video_list = get_video_list(args.test_dir)
        for idx, video_file in enumerate(video_list):
            video_name = os.path.basename(video_file)

            videoReader = MediaReader(video_file, 32, 1)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以使用'XVID'或'H264'等其他编码器
            outVideo = cv2.VideoWriter(args.save_path, fourcc, videoReader.frame_rate, (videoReader.frame_width, videoReader.frame_height))

            while not videoReader.end:
                _, frame = videoReader.read()

                if (len(videoReader.stack)) == videoReader.win_size or videoReader.end:
                    video = np.array(videoReader.stack)

                    # TODO: 
                    output = vecpred.predict(video, [644, 446, 1633, 1424])

                    action_info = StateMachine.checkState(output['obj_class'])

                    print(output['obj_class'], output['cosine'], action_info)

                    videoReader.pop()
                    if action_info:
                        cv2.putText(frame, "Action: " + str(output['obj_class']), (20, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, "Action Error", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3)
                
                outVideo.write(frame)
            
            videoReader.release()
            outVideo.release()


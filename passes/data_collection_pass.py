from passes.base_pass import *
import cv2


class DataCollectionPass(BasePass):
    def __init__(self, video_data, frames):
        super().__init__()
        self.video_data = video_data
        self.video = None
        self.frames = frames

    def execute(self):
        super().execute()
        self.video = cv2.VideoCapture(self.video_data.in_path)
        self.video_data.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_data.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_data.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_data.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        print("----Input Video----")
        print("path:\t\t\t\t{}".format(self.video_data.in_path))
        print("dimensions:\t\t\t({} x {})".format(self.video_data.width, self.video_data.height))
        print("video length:\t\t{} seconds".format(self.video_data.frame_count / self.video_data.fps))
        print("fps:\t\t\t\t{}".format(self.video_data.fps))
        print("number of frames:\t{}".format(self.video_data.frame_count))
        print("-------------------\n")

    def get_values(self):
        return self.video_data, self.frames


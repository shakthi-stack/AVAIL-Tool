import glob

from passes.base_pass import *
from util.objects import *

import os
import cv2
import sys
import numpy as np

from passes.data_collection_pass import DataCollectionPass
from util.dfl_location import DFL_absolute_path
from pathlib import PurePath

sys.path.insert(1, DFL_absolute_path)
# if DFL's path is correctly inserted, these imports should work
try:
    from mainscripts.Extractor import ExtractSubprocessor
    from core.leras.nn import nn
    from facelib.FaceType import FaceType
except:
    print("Something went wrong finding DeepFaceLab co-directory's modules")


class DataCollectionDeepFaceLab(DataCollectionPass):
    def __init__(self, video_data, frames, frames_dir="frames_dir", frame_skip=0, image_scale=0.5):
        super().__init__(video_data, frames)
        self.video_data = video_data
        self.frames = frames
        self.frames_dir = frames_dir
        self.frame_skip = max(0, frame_skip)
        self.image_scale = image_scale
        self.data = []

        self.frames_copied = False
        nn.initialize_main_env()

    def check_frames_dir(self):
        try:
            os.makedirs(self.frames_dir, exist_ok=True)

            frame_name = "{:05d}.png".format(0)
            file_name = self.frames_dir + "/" + frame_name
            img_first = cv2.imread(file_name)

            last_frame = self.video_data.frame_count - 1
            if self.frame_skip is not 0:
                last_frame = 0
                while last_frame + self.frame_skip + 1 < self.video_data.frame_count:
                    last_frame += self.frame_skip + 1

            frame_name = "{:05d}.png".format(last_frame)
            file_name = self.frames_dir + "/" + frame_name
            img_last = cv2.imread(file_name)

            video = cv2.VideoCapture(self.video_data.in_path)
            _, real_first = video.read()
            video.set(1, last_frame)
            _, real_last = video.read()
            video.release()
            if np.array_equal(img_first, real_first) and np.array_equal(img_last, real_last):
                return True
            print("Frames directory doesn't contain this video's frames.  They will be extracted")
        except:
            print("Frames directory doesn't contain this video's frames.  They will be extracted")
        return False

    def empty_frames_dir(self):
        files = glob.glob(self.frames_dir + "/*.png")
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def execute(self):
        super().execute()
        if not os.path.isdir(self.frames_dir):
            os.mkdir(self.frames_dir)

        self.video = cv2.VideoCapture(self.video_data.in_path)

        index = 0
        next_tick = 0
        tick_interval = 0.05
        if self.check_frames_dir():
            self.frames_copied = True
        else:
            self.empty_frames_dir()
            print("Copying video frames into temporary directory")
            if self.frame_skip is not 0:
                print("\tSkipping every {0} frames".format(self.frame_skip))
        while self.video.isOpened() and index < self.video_data.frame_count:
            self.iterative_step(self.video, index)
            # quick command line progress bar
            if float(index) / self.video_data.frame_count > next_tick:
                next_tick += tick_interval
                print("{0:2.0f}% ---- frame {1:5.0f} ---- time {2:10.4f} s".format(
                    float(index) / self.video_data.frame_count * 100,
                    index,
                    time.time() - self.start_time))

            index += 1
            for x in range(self.frame_skip):
                if not self.frames_copied:
                    _, _ = self.video.read()
                index += 1

        print("Running DeepFaceLab s3fd Face Detector")
        extract = ExtractSubprocessor(self.data, 'rects-s3fd', img_scale=self.image_scale, jpeg_quality=90, face_type=FaceType.WHOLE_FACE, device_config=nn.DeviceConfig.GPUIndexes([0]))
        self.data = extract.run()

        index = 0
        next_tick = 0
        tick_interval = 0.05
        print("Copying collected data into serializable format")
        for d in self.data:
            frame = FrameData(index)
            for entry in d.rects:
                face = Face(entry[0], entry[1], entry[2] - entry[0], entry[3] - entry[1])
                face.bind_to_frame(self.video_data.width, self.video_data.height)
                frame.add_face(face)
            self.frames.append(frame)
            index += 1
            for x in range(self.frame_skip):
                self.frames.append(FrameData(index))
                index += 1

    def iterative_step(self, video, index):
        # setup the frame object and read the image
        try:
            img = None
            if not self.frames_copied:
                _, img = video.read()
            frame_name = "{:05d}.png".format(index)
            file_name = self.frames_dir + "/" + frame_name
            if not self.frames_copied:
                cv2.imwrite(file_name, img)

            self.data.append(ExtractSubprocessor.Data(PurePath(file_name)))
        except:
            # keep running without this frame
            a = 1


    def get_values(self):
        return self.video_data, self.frames



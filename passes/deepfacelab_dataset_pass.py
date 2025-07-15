import glob

from passes.base_pass import *

import os
import cv2
import sys
import shutil

from util.dfl_location import DFL_absolute_path

sys.path.insert(1, DFL_absolute_path)

from pathlib import PurePath

try:
    from mainscripts.Extractor import ExtractSubprocessor
    from core.leras.nn import nn
    from facelib.FaceType import FaceType
except:
    print("Something went wrong finding DeepFaceLab co-directory's modules")


class DeepFaceLabTrainingPass():
    def __init__(self, in_paths, out_path, cull=10):
        super().__init__()
        self.in_paths = in_paths
        self.out_path = out_path
        self.cull = cull

    def execute(self):
        if not os.path.isdir(self.out_path):
            os.mkdir(self.out_path)

        idx = 0
        for dir in self.in_paths:
            print("Copying from {0}".format(dir))
            in_files = glob.glob(dir + "/*.jpg", recursive=True)
            cull_count = 0
            for file in in_files:
                if cull_count == self.cull:
                    out = self.out_path + "/{0}.jpg".format(idx)
                    idx += 1
                    shutil.copy2(file, out)
                    cull_count = 0
                else:
                    cull_count += 1


class DeepFaceLabDatasetPass(BasePass):
    def __init__(self, video_data, frames, face_res=256, dataset_path="dst", write_frames=True, write_align=True):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.face_res = face_res
        self.dataset_path = dataset_path
        self.aligned_path = self.dataset_path + "//aligned"
        self.write_frames = write_frames
        self.write_align = write_align
        self.data = []
        nn.initialize_main_env()

    def execute(self):
        super().execute()

        if not os.path.isdir(self.dataset_path):
            os.mkdir(self.dataset_path)
        if not os.path.isdir(self.aligned_path):
            os.mkdir(self.aligned_path)

        self.video = cv2.VideoCapture(self.video_data.in_path)

        index = 0
        next_tick = 0
        tick_interval = 0.05
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
        if self.write_align:
            extract = ExtractSubprocessor(self.data, 'landmarks-final', image_size=self.face_res, jpeg_quality=90,
                                          face_type=FaceType.WHOLE_FACE, final_output_path=PurePath(self.aligned_path),
                                          device_config=nn.DeviceConfig.GPUIndexes([0]))
            extract.run()

    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame_data = self.frames[index]
        frame_name = "{:05d}.jpg".format(index)
        file_name = self.dataset_path + "/" + frame_name
        if self.write_frames:
            cv2.imwrite(file_name, img)

        idx = 0
        detected_faces = []
        for face in frame_data.faces:
            if face.tag == "not child":
                continue
            x = face.x
            y = face.y
            w = face.w
            h = face.h
            detected_faces.append((x, y, x + w, y + h))
            idx += 1
        self.data.append(ExtractSubprocessor.Data(PurePath(file_name), rects=detected_faces))

    def get_values(self):
        return self.video_data, self.frames

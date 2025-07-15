from util.objects import *
from passes.data_collection_pass import DataCollectionPass
import cv2
import time


class CascadeBaseClass(DataCollectionPass):
    def __init__(self, video_data, frames,
                 face_cascade=cv2.CascadeClassifier('haar/haarcascade_face.xml'),
                 eye_cascade=cv2.CascadeClassifier('haar/haarcascade_eye.xml')):
        super().__init__(video_data, frames)
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade

    def execute(self):
        super().execute()

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

    def iterative_step(self, video, index):
        print("CascadeBaseClass: iterative_step not implemented")
        return


class DataCollectionFaceCascade(CascadeBaseClass):
    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame = FrameData(index)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces and add to frame data
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            frame.add_face(Face(x, y, w, h))
        # store this frame
        self.frames.append(frame)


class DataCollectionFaceAndEyeCascades(CascadeBaseClass):
    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame = FrameData(index)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces and add to frame data
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            frame.add_face(Face(x, y, w, h))
        # detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in eyes:
            frame.add_eye(Eye(x, y, w, h))
        # store this frame
        self.frames.append(frame)


class DataCollectionCascadesWithEyeCulling(CascadeBaseClass):
    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame = FrameData(index)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces and add to frame data
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            frame.add_face(Face(x, y, w, h))
        # detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in eyes:
            frame.add_eye(Eye(x, y, w, h))
        frame.cull_eyes()
        # store this frame
        self.frames.append(frame)

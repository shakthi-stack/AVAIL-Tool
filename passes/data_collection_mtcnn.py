from util.objects import *
from passes.data_collection_pass import DataCollectionPass
import mtcnn
import time
import threading
import os

cores_to_use = int(os.cpu_count() - 1)


class DataCollectionMTCNN(DataCollectionPass):
    def __init__(self, video_data, frames):
        super().__init__(video_data, frames)
        self.detector = mtcnn.mtcnn.MTCNN()

    def execute(self):
        super().execute()

        index = 0
        next_tick = 0
        tick_interval = 0.05
        while self.video.isOpened() and index < self.video_data.frame_count:
            start_time = time.time()
            self.iterative_step(self.video, index)
            # print("{0} s".format(time.time() - start_time))
            # quick command line progress bar
            if float(index) / self.video_data.frame_count > next_tick:
                next_tick += tick_interval
                print("{0:2.0f}% ---- frame {1:5.0f} ---- time {2:10.4f} s".format(
                    float(index) / self.video_data.frame_count * 100,
                    index,
                    time.time() - self.start_time))

            index += 1

    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame = FrameData(index)

        faces = self.detector.detect_faces(img)
        for face in faces:
            x = face["box"][0]
            y = face["box"][1]
            w = face["box"][2]
            h = face["box"][3]
            face_obj = Face(x, y, w, h, face["keypoints"])
            face_obj.bind_to_frame(self.video_data.width, self.video_data.height)
            frame.add_face(face_obj)
        # store this frame
        self.frames.append(frame)


class DataCollectionMTCNNMultithread(DataCollectionPass):
    def __init__(self, video_data, frames, batch_size=cores_to_use):
        super().__init__(video_data, frames)
        # self.detector = mtcnn.mtcnn.MTCNN()
        self.batch = batch_size

        self.detectors = []
        for x in range(self.batch):
            self.detectors.append(mtcnn.mtcnn.MTCNN())

    def execute(self):
        super().execute()
        for x in range(self.video_data.frame_count):
            self.frames.append(None)
        threads = [None for x in range(self.batch)]

        index = 0
        next_tick = 0
        tick_interval = 0.05
        while self.video.isOpened() and index < self.video_data.frame_count:
            start_time = time.time()
            for x in range(self.batch):
                if index == self.video_data.frame_count:
                    continue
                _, img = self.video.read()
                threads[x] = MTCNNThread(self, x + 1, img, index)
                threads[x].start()
                index += 1
            for x in range(self.batch):
                threads[x].join()
                self.frames[threads[x].index] = threads[x].frame
            # print("{0} s".format((time.time() - start_time) / cores_to_use))
            # quick command line progress bar
            if float(index) / self.video_data.frame_count > next_tick:
                next_tick += tick_interval
                print("{0:2.0f}% ---- frame {1:5.0f} ---- time {2:10.4f} s".format(
                    float(index) / self.video_data.frame_count * 100,
                    index,
                    time.time() - self.start_time))
                print("Last batch of {0}, {1} s".format(cores_to_use, (time.time() - start_time) / cores_to_use))


class MTCNNThread(threading.Thread):
    def __init__(self, obj, threadID, img, index):
        threading.Thread.__init__(self)
        self.x = threadID - 1
        self.threadID = str(threadID)
        self.obj = obj
        self.img = img
        self.index = index
        self.frame = None

    def run(self):
        # print("Starting " + self.threadID)
        self.frame = iterative_step_multi(self.obj, self.img, self.index, self.x)
        # print("Exiting " + self.threadID)


def iterative_step_multi(obj, img, index, x):
    # setup the frame object and read the image
    frame = FrameData(index)
    try:
        faces = obj.detectors[x].detect_faces(img)
        # faces = obj.detector.detect_faces(img)
    except:
        print("Detector failure at frame {0}".format(index))
        return frame

    for face in faces:
        x = face["box"][0]
        y = face["box"][1]
        w = face["box"][2]
        h = face["box"][3]
        face_obj = Face(x, y, w, h, face["keypoints"])
        face_obj.bind_to_frame(obj.video_data.width, obj.video_data.height)
        frame.add_face(face_obj)
    # store this frame
    return frame

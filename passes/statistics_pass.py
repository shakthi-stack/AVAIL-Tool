from passes.base_pass import *

import os
import csv


class StatisticsPass(BasePass):
    def __init__(self, video_data, frames, csv_filepath=None):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.csv_filepath = csv_filepath

    def execute(self):
        super().execute()

        if self.csv_filepath is not None and not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Clip', '# Frames', '# Detected Faces', '# Face Chains', '# Frames w 1 Detected Face', '% Frames w 1 Detected Face', '# Frames w 2 Detected Faces', '% Frames w 2 Detected Faces', '# Frames w 3+ Detected Faces', '% Frames w 3+ Detected Faces'])

        # Want to collect info on
        # number of faces detected
        num_faces = 0
        # number of frames with a face
        frames_w_face = 0
        # number of frames with multiple faces
        frames_w_twoface = 0
        frames_w_threeface = 0
        # number of face chains
        num_chains = 0
        index = 0
        while index < len(self.frames):
            frame = self.frames[index]
            if frame is None:
                index += 1
                continue
            if len(frame.faces) != 0:
                frames_w_face += 1
            if len(frame.faces) > 1:
                frames_w_twoface += 1
            if len(frame.faces) > 2:
                frames_w_threeface += 1
            num_faces += len(frame.faces)
            for face in frame.faces:
                if face.backward_neighbor is None:
                    num_chains += 1
            index += 1

        num_frames = len(self.frames)
        perc_one_face = frames_w_face / num_frames * 100
        perc_two_face = frames_w_twoface / num_frames * 100
        perc_three_face = frames_w_threeface / num_frames * 100
        print("\n----FACE DETECTION STATISTICS----")
        print("Frame count:\t{}".format(num_frames))
        print("--------------------------------")
        print("Number of faces detected:\t{}".format(num_faces))
        print("--------------------------------")
        print("Number of frames with at least one face:\t{}".format(frames_w_face))
        print("Percent of frames with at least one face:\t{0:2.0f}%".format(perc_one_face))
        print("--------------------------------")
        print("Number of frames with two or more faces:\t{}".format(frames_w_twoface))
        print("Percent of frames with two or more faces:\t{0:2.0f}%".format(perc_two_face))
        print("--------------------------------")
        print("Number of frames with three or more faces:\t{}".format(frames_w_threeface))
        print("Percent of frames with three or more faces:\t{0:2.0f}%".format(perc_three_face))
        print("--------------------------------")
        print("Number of face chains:\t{}".format(num_chains))
        print("--------------------------------\n")

        if self.csv_filepath is not None:
            with open(self.csv_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.video_data.out_path.split('/')[-1], num_frames, num_faces, num_chains, frames_w_face, perc_one_face, frames_w_twoface, perc_two_face, frames_w_threeface, perc_three_face])

    def get_values(self):
        return self.video_data, self.frames


class HeuristicStatisticsPass(BasePass):
    def __init__(self, video_data, frames, csv_filepath=None):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.csv_filepath = csv_filepath

    def execute(self):
        super().execute()

        if not hasattr(self.frames[0], 'manual_annotation') or self.frames[0].manual_annotation is None:
            print("Chain correction hasn't been run on {0}".format(self.video_data.in_path))
            return

        if self.csv_filepath is not None and not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Clip', '# Frames', '# Manually Annotated Frames', '% Annotated', '# Detected Child Faces', '% Child/Annotated', '# Detected Child Faces (with Heuristic)', '% Heuristic/Annotated'])

        # Want to collect info on
        # number of faces detected
        num_annotated = 0
        num_child = 0
        num_heuristic = 0
        # number of face chains
        num_chains = 0
        index = 0
        while index < len(self.frames):
            frame = self.frames[index]
            if frame is None:
                index += 1
                continue
            if frame.manual_annotation.has_face:
                num_annotated += 1
            for f in frame.faces:
                if f.tag is None:
                    num_child += 1
                    num_heuristic += 1
                if f.tag == 'heuristic':
                    num_heuristic += 1
            index += 1

        num_frames = len(self.frames)
        perc_annotated = num_annotated / num_frames * 100
        perc_child = num_child / num_annotated * 100
        perc_heuristic = num_heuristic / num_frames * 100

        if self.csv_filepath is not None:
            with open(self.csv_filepath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.video_data.out_path.split('/')[-1], num_frames, num_annotated, perc_annotated, num_child, perc_child, num_heuristic, perc_heuristic])

    def get_values(self):
        return self.video_data, self.frames

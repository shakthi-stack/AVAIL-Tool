import sys

from passes.data_collection_pass import *

import pickle
sys.path.insert(1, "../manual_annotation/")


class PickleEncodePass(BasePass):
    def __init__(self, video_data, frames, path=""):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.path = path

    def execute(self):
        super().execute()

        outfile = open(self.path, "wb")
        pickle.dump(self.frames, outfile)
        outfile.close()

    def get_values(self):
        return self.video_data, self.frames


class PickleDecodePass(DataCollectionPass):
    def __init__(self, video_data, frames, path=""):
        super().__init__(video_data, frames)
        self.path = path

    def execute(self):
        super().execute()

        print("Reading serialized data from {}".format(self.path))
        infile = open(self.path, "rb")
        loaded_frames = pickle.load(infile)
        for f in loaded_frames:
            self.frames.append(f)
        infile.close()

    def get_values(self):
        return self.video_data, self.frames


import ntpath
import os

from util.objects import VideoData


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def process_video(in_path, out_path, pickle_path, processor, skip_pickled=False,
                  only_unpickle=False):
    video_data = VideoData(in_path, out_path)
    frames = []

    if skip_pickled and os.path.exists(pickle_path):
        print("Skipping {0}, file has already been pickled.".format(in_path))
        return
    if only_unpickle and not os.path.exists(pickle_path):
        print("Skipping {0}, file has no pickled information.".format(in_path))
        return

    processor.process()

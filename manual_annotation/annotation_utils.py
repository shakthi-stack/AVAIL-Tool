import glob
import ntpath
import os
import time

from tqdm import tqdm
import cv2
import numpy as np
from util.objects import ManualAnnotation
from util.util import path_leaf

# import ffmpeg


def populate_annotations(path, dir="frames/"):
    annotations = []
    framesdir = dir + path_leaf(path[:-4]) + "/"
    video = cv2.VideoCapture(path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0
    while idx < num_frames:
        framespath = framesdir + "{0}.jpg".format(idx)
        annotations.append(ManualAnnotation(framespath))
        idx += 1
    return annotations


def resize_dims(dims, scale):
    width = int(dims[1] * scale)
    height = int(dims[0] * scale)
    dim = (width, height)
    return dim


def create_dir(path, annotations, dir="frames/", scale=1):
    if check_dir(path, dir=dir, scale=scale):
        return
    dir = dir + path_leaf(path[:-4]) + "/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    video = cv2.VideoCapture(path)
    dims = video.get(cv2.CAP_PROP_FRAME_HEIGHT), video.get(cv2.CAP_PROP_FRAME_WIDTH)
    width, height = resize_dims(dims, scale)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in tqdm(range(length)):
        valid, img = video.read()
        if valid:
            last_img = img
        if not valid:
            # use the last working image if it's broken
            img = last_img
        try:
            img = cv2.resize(img, resize_dims(img.shape[:2], scale))
            cv2.imwrite(annotations[idx].path, img)
        except:
            # end of video, break the loop
            break

    # start_time = time.time()
    # print(f'Processing {path}.')
    # ffmpeg.input(path).filter('scale', width, height).\
    #     output(f'{dir}/%d.jpg', start_number=0).overwrite_output().run(quiet=True)
    #
    # # writing the first and last frame over in opencv
    # video = cv2.VideoCapture(path)
    # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # last_frame = num_frames - 1
    #
    # _, rf = video.read()
    # real_first = cv2.resize(rf, (width, height))
    # cv2.imwrite(f'{dir}/0.jpg', real_first)
    # video.set(1, last_frame)
    # _, rl = video.read()
    # real_last = cv2.resize(rl, (width, height))
    # cv2.imwrite(f'{dir}/{last_frame}.jpg', real_last)
    # video.release()
    #
    # print(f'time: {time.time() - start_time} s')


def check_dir(path, dir="frames/", scale=1):
    try:
        video = cv2.VideoCapture(path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        last_frame = num_frames - 1

        dir = dir + path_leaf(path[:-4]) + "/"
        print("checking {0}".format(dir))
        frame_name = "{0}.jpg".format(0)
        file_name = dir + frame_name
        img_first = cv2.imread(file_name)
        frame_name = "{0}.jpg".format(last_frame)
        file_name = dir + frame_name
        img_last = cv2.imread(file_name)

        _, rf = video.read()
        real_first = cv2.resize(rf, resize_dims(rf.shape[:2], scale))
        video.set(1, last_frame)
        _, rl = video.read()
        real_last = cv2.resize(rl, resize_dims(rf.shape[:2], scale))
        video.release()

        # this line checks that the video and directory's first and last frames match up,
        # accounting for video compression
        if np.allclose(img_first, real_first, rtol=50, atol=50) and np.allclose(img_last, real_last, rtol=50, atol=50):
            return True
        else:
            print("first: {0}, last: {1}".format(np.allclose(img_first, real_first, rtol=30, atol=20),
                                                 np.allclose(img_last, real_last, rtol=30, atol=20)))
        print(f"Frames directory doesn't contain {path}\'s frames.  They will be extracted.")
    except:
        print(f"Frames directory doesn't contain {path}\'s frames.  They will be extracted.")
    return False


def set_annotations(annotations):
    is_active = False
    for x in range(len(annotations)):
        annot = annotations[x]
        if annot.end:
            is_active = False
        if annot.start:
            is_active = True
        annot.has_face = annot.end or annot.start or is_active
        annotations[x] = annot
    return annotations


def create_annotation_directory(in_path, dir, scale=1):
    if os.path.isdir(in_path):
        in_files = glob.glob(in_path + "/*.mp4", recursive=False)
        for file in in_files:
            annotations = populate_annotations(file, dir=dir)
            create_dir(file, annotations, dir=dir, scale=scale)
    elif os.path.isfile(in_path):
        annotations = populate_annotations(in_path, dir=dir)
        create_dir(in_path, annotations, dir=dir, scale=scale)

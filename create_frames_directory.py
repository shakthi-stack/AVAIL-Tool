import argparse
import glob
import os

from manual_annotation.annotation_utils import create_annotation_directory
from util.util import path_leaf

parser = argparse.ArgumentParser(description='The manual annotation system uses a temporary directory of frames '
                                             'extracted from the source videos.  It can take a while to run depending '
                                             'on the number of videos and their durations.  You can call this file to '
                                             'create a full frames directory rather than for each file when the GUI is '
                                             'first run.')
parser.add_argument('input_path', type=str,
                    help='path to input file or directory.  note: accepts .mp4, .avi, and .mov files.')
parser.add_argument('frames_dir', type=str,
                    help='directory where video frames are stored.')
parser.add_argument('-s', '--scale_factor', type=float, default=1,
                    help='scale factor for extracted video frames displayed in the GUI.  For example, a value of 1.5 '
                         'would be 50% larger than the video\'s resolution, a value of 0.7 would be 30% smaller.')

args = parser.parse_args()


def create_frames_directory(in_path):
    create_annotation_directory(in_path, args.frames_dir, scale=args.scale_factor)


if __name__ == '__main__':
    if args.frames_dir[-1] != '/':
        args.frames_dir += '/'
    if not os.path.exists(args.frames_dir):
        response = input(f'frames directory {args.frames_dir} does not exist.  Create directory? [y/n]')
        if response in ['y', 'yes', 'YES', 'Y']:
            os.makedirs(args.frames_dir, exist_ok=True)
        else:
            print('terminating program.')
            exit()

    if os.path.isdir(args.input_path):
        print(f'will process all files in {args.input_path}.')
        in_files = glob.glob(f'{args.input_path}//*')
        for in_file in in_files:
            if path_leaf(in_file).split(".")[1] not in ['mp4', 'avi', 'mov']:
                continue
            create_frames_directory(in_file)

    elif os.path.isfile(args.input_path):
        print(f'will process {args.input_path}.')
        create_frames_directory(args.input_path)


    else:
        raise Exception(f'directory path mismatch between input and output arguments.')

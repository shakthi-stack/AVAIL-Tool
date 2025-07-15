import argparse
import glob
import os

from manual_annotation.chain_corrections import run_chain_corrections
from manual_annotation.manual_annotator import run_manual_annotator
from passes.heuristic_pass import HeuristicPass
from passes.interpolation_pass import CurveFittingPass
from passes.pickle_pass import PickleDecodePass, PickleEncodePass
from passes.statistics_pass import StatisticsPass
from util.objects import VideoData
from util.util import path_leaf
from util.video_processor import VideoProcessor

parser = argparse.ArgumentParser(description='manually annotate over detected frames, discard unwanted subjects, '
                                             'and interpolate between frames.')
parser.add_argument('input_path', type=str,
                    help='path to input file or directory.  note: accepts .mp4, .avi, and .mov files.')
parser.add_argument('pickle_path', type=str,
                    help='path to input file/directory\'s matching encoded pickle data.')
parser.add_argument('frames_dir', type=str,
                    help='directory where video frames are stored.')
parser.add_argument('-s', '--scale_factor', type=float, default=1,
                    help='scale factor for extracted video frames displayed in the GUI.  For example, a value of 1.5 '
                         'would be 50% larger than the video\'s resolution, a value of 0.7 would be 30% smaller.')

args = parser.parse_args()


def manual_annotation(in_path, pickle_path):
    run_manual_annotator(in_path, args.frames_dir, args.scale_factor)
    run_chain_corrections(in_path, pickle_path, args.scale_factor)

    video_data = VideoData(in_path, "")
    frames = []

    processor = VideoProcessor([
        PickleDecodePass(video_data, frames, path=pickle_path),
        HeuristicPass(video_data, frames, max_size=1.1, max_offset=10000),
        PickleEncodePass(video_data, frames, path=pickle_path),
        StatisticsPass(video_data, frames),
    ])
    processor.process()


if __name__ == '__main__':
    if args.frames_dir[-1] != '/':
        args.frames_dir += '/'
    if not os.path.exists(args.frames_dir):
        response = input(f'frames directory {args.frames_dir} does not exist.  Create directory? [y/n]')
        if response in ['y', 'yes', 'YES', 'Y']:
            os.makedirs(args.frames_dir, exist_ok=True)
        else:
            exit()

    if os.path.isdir(args.input_path) and os.path.isdir(args.pickle_path):
        print(f'will process all files sequentially in {args.input_path}/{args.pickle_path}.')
        in_files = glob.glob(f'{args.input_path}//*')
        for in_file in in_files:
            if path_leaf(in_file).split(".")[1] not in ['mp4', 'avi', 'mov']:
                continue
            pickle_file = f'{args.pickle_path}//{path_leaf(in_file).split(".")[0]}.pickle'
            manual_annotation(in_file, pickle_file)

    elif os.path.isfile(args.input_path) and os.path.isfile(args.pickle_path):
        print(f'will process {args.input_path}/{args.pickle_path}.')
        manual_annotation(args.input_path, args.pickle_path)

    else:
        raise Exception(f'directory path mismatch between input and output arguments.')

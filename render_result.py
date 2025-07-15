import argparse
import glob
import os

from passes.pickle_pass import PickleDecodePass
from passes.rendering_pass import RenderingPassBlurChild, RenderingPassBlur, RenderingPassBlurEverything, \
    RenderingPassBoxes, RenderingPassNearestNeighbor, RenderingPassKeypoints, MultiRenderingPass
from passes.statistics_pass import StatisticsPass
from util.util import path_leaf

from util.objects import VideoData
from util.video_processor import VideoProcessor

parser = argparse.ArgumentParser(description='Renders debug/blur results on videos that have encoded '
                                             'face information.')
parser.add_argument('input_path', type=str,
                    help='path to input file or directory.  note: accepts .mp4, .avi, and .mov files.')
parser.add_argument('output_path', type=str,
                    help='file or directory to store rendered results.')
parser.add_argument('pickle_path', type=str,
                    help='file or directory where encoded pickle data should be stored.')
parser.add_argument('-b', '--blur', type=str, choices=['none', 'subject_face', 'all_faces', 'whole_frame'], default='none',
                    help='determine which areas, if any, to blur.')
parser.add_argument('-i', '--intensity', type=float, default=20,
                    help='blur strength.  Lower values are stronger.  Default 20.')
parser.add_argument('-d', '--debug', type=str, choices=['none', 'boxes', 'neighbors', 'boxes+neighbors', 'keypoints'],
                    default='none', help='determine any debug commands.  note: keypoints requires mtcnn.')

args = parser.parse_args()


def create_processor(video_data, frames, pickle_path):
    render_passes = []
    print(args.intensity)
    if args.blur == 'subject_face':
        render_passes.append(RenderingPassBlurChild(video_data, frames, args.intensity))
    if args.blur == 'all_faces':
        render_passes.append(RenderingPassBlur(video_data, frames, args.intensity))
    if args.blur == 'whole_frame':
        render_passes.append(RenderingPassBlurEverything(video_data, frames, args.intensity))

    if 'boxes' in args.debug:
        render_passes.append(RenderingPassBoxes(video_data, frames))
    if 'neighbors' in args.debug:
        render_passes.append(RenderingPassNearestNeighbor(video_data, frames))
    if args.debug == 'keypoints':
        render_passes.append(RenderingPassKeypoints(video_data, frames))

    return VideoProcessor([
        PickleDecodePass(video_data, frames, path=pickle_path),
        MultiRenderingPass(render_passes, video_data, frames),
        StatisticsPass(video_data, frames),
    ])


if __name__ == '__main__':
    if os.path.isdir(args.input_path) and os.path.isdir(args.output_path) and os.path.isdir(args.pickle_path):
        print(f'will process all files in {args.input_path} to {args.output_path}.')
        in_files = glob.glob(f'{args.input_path}//*')
        for in_file in in_files:
            if path_leaf(in_file).split(".")[1] not in ['mp4', 'avi', 'mov']:
                continue
            pickle_file = f'{args.output_path}//{path_leaf(in_file).split(".")[0]}.pickle'
            out_file = f'{args.output_path}//{path_leaf(in_file)}'

            create_processor(VideoData(in_file, out_file), [], pickle_file).process()

    elif os.path.isfile(args.input_path) and os.path.isfile(args.pickle_path):
        print(f'will process {args.input_path} to {args.output_path}.')

        create_processor(VideoData(args.input_path, args.output_path), [], args.pickle_path).process()

    else:
        raise Exception(f'directory path mismatch between input and output arguments.')

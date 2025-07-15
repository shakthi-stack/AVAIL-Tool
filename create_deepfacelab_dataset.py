import argparse
import glob
import os

from passes.interpolation_pass import FaceInterpPass
from passes.nearest_neighbor_pass import NearestNeighborPass, RemoveNeighborsPass
from passes.pickle_pass import PickleEncodePass, PickleDecodePass
from passes.statistics_pass import StatisticsPass
from util.dfl_location import DFL_absolute_path
from util.util import path_leaf

if DFL_absolute_path:
    from passes.data_collection_deepfacelab import DataCollectionDeepFaceLab
    from passes.deepfacelab_dataset_pass import DeepFaceLabDatasetPass

from util.objects import VideoData
from util.video_processor import VideoProcessor

parser = argparse.ArgumentParser(description='Can create a DeepFaceLab compatible face directory using manually '
                                             'annotated frames.  Requires custom DFL co-directory to be set in '
                                             'util/dfl_location.py.  Not recommended.')
parser.add_argument('input_path', type=str,
                    help='path to input file or directory.  note: accepts .mp4, .avi, and .mov files.')
parser.add_argument('frames_path', type=str,
                    help='directory where output directories are stored.')
parser.add_argument('pickle_path', type=str,
                    help='file or directory where encoded pickle data is stored.')
parser.add_argument('-d', '--detector_scale', type=int, default=0.5,
                    help='amount to scale the video frames before DFL s3fd processing.')
parser.add_argument('-f', '--frame_skip', type=int,
                    help='optionally perform interpolation, detecting a face every F frames and interpolating '
                         'the results.')
parser.add_argument('-s', '--skip_processed_files', action='store_true',
                    help='optionally skip files that already have output frames.  useful for resuming the '
                         'program when processing large directories.')

args = parser.parse_args()


def create_processor(video_data, frames, pickle_path):

    try:
        VideoProcessor([
            PickleDecodePass(video_data, frames, path=pickle_path),
            # Short range interpolation tied to the DataCollectionDeepFaceLab's frame_skip
            NearestNeighborPass(video_data, frames, search_depth=args.frame_skip + 1),
            FaceInterpPass(video_data, frames),
            RemoveNeighborsPass(video_data, frames),
            PickleEncodePass(video_data, frames, path=pickle_path),
            DeepFaceLabDatasetPass(video_data, frames, dataset_path=video_data.out_path),
            StatisticsPass(video_data, frames),
        ]).process()
    except:
        print(f'No pickle data for {video_data.in_path}, will attempt to collect data.')
        VideoProcessor([
            DataCollectionDeepFaceLab(video_data, frames, frame_skip=args.frame_skip, image_scale=args.detector_scale),
            NearestNeighborPass(video_data, frames, search_depth=args.frame_skip + 1),
            FaceInterpPass(video_data, frames),
            RemoveNeighborsPass(video_data, frames),
            PickleEncodePass(video_data, frames, path=pickle_path),
            DeepFaceLabDatasetPass(video_data, frames, dataset_path=video_data.out_path),
            StatisticsPass(video_data, frames),
        ]).process()


if __name__ == '__main__':
    if os.path.isdir(args.input_path) and os.path.isdir(args.pickle_path) and os.path.isdir(args.frames_path):
        print(f'will process all files in {args.input_path} to {args.frames_path}.')
        in_files = glob.glob(f'{args.input_path}//*')
        for in_file in in_files:
            if path_leaf(in_file).split(".")[1] not in ['mp4', 'avi', 'mov']:
                continue
            pickle_file = f'{args.pickle_path}//{path_leaf(in_file).split(".")[0]}.pickle'
            out_file = f'{args.frames_path}//{path_leaf(in_file).split(".")[0]}/'
            if args.skip_processed_files and os.path.exists(pickle_file):
                print("Skipping {0}, file has already been pickled.".format(in_file))
            else:
                create_processor(VideoData(in_file, out_file), [], pickle_file)

    elif os.path.isfile(args.input_path) and os.path.isdir(args.frames_path):
        print(f'will process {args.input_path} to {args.frames_path}.')
        if args.skip_processed_files and os.path.exists(args.pickle_path):
            print("Skipping {0}, file has already been pickled.".format(args.input_path))
        else:
            create_processor(VideoData(args.input_path, args.frames_path), [], args.pickle_path)

    else:
        raise Exception(f'directory path mismatch between input and pickle arguments, or frames path '
                        f'is not a directory.')

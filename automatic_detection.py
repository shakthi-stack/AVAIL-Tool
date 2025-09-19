import argparse
import glob
import os

from passes.interpolation_pass import FaceInterpPass
from passes.nearest_neighbor_pass import NearestNeighborPass, RemoveNeighborsPass
from passes.pickle_pass import PickleEncodePass
from passes.statistics_pass import StatisticsPass
from util.dfl_location import DFL_absolute_path
from util.util import path_leaf

if DFL_absolute_path:
    from passes.data_collection_deepfacelab import DataCollectionDeepFaceLab

from util.objects import VideoData
from util.video_processor import VideoProcessor

def _parse_wh(s):
    try:
        w, h = s.lower().split('x')
        return (int(w), int(h))
    except Exception:
        raise argparse.ArgumentTypeError(f"Bad --scrfd_size '{s}', expected WxH (e.g., 640x640)")


parser = argparse.ArgumentParser(description='runs automatic face detection on a video or directory of videos.')
parser.add_argument('input_path', type=str,
                    help='path to input file or directory.  note: accepts .mp4, .avi, and .mov files.')
parser.add_argument('output_path', type=str,
                    help='file or directory where encoded pickle data should be stored.')
parser.add_argument('-d', '--detection_method', type=str, choices=['s3fd', 'haar', 'mtcnn', 'dfl','yunet','scrfd'], default='s3fd',
                    help='which face detection method to apply.  Default is s3fd.')
parser.add_argument('--yunet_model', type=str, default='models/face_detection_yunet_2023mar.onnx',
                     help='path to YuNet ONNX model (FaceDetectorYN).')
parser.add_argument('--scrfd_onnx', default='models/scrfd_10g_bnkps.onnx',
                    help='Path to SCRFD ONNX file (local, no downloads).')
parser.add_argument('--scrfd_size', type=_parse_wh, default=(640, 640),
                    help='SCRFD input WxH (must be divisible by 32).')
parser.add_argument('--scrfd_thresh', type=float, default=0.5,
                    help='SCRFD detection threshold.')
parser.add_argument('--scrfd_providers', nargs='+',
                    default=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                   # default=['CPUExecutionProvider'],
                    help='ONNX Runtime providers priority list.')
parser.add_argument('-i', '--interpolation', type=int,
                    help='optionally perform interpolation at this stage, filling in the specified number of frames.')
parser.add_argument('-s', '--skip_processed_files', action='store_true',
                    help='optionally skip files that already have encoded information.  useful for resuming the '
                         'program when processing large directories.')
parser.add_argument('--device', default='cuda',
                    help='pytorch device to use (when running s3fd detection.  note: MTCNN uses keras so will '
                         'search for GPU using cudnn.)')

args = parser.parse_args()


def create_processor(video_data, frames):
    if args.detection_method == 's3fd':
        from passes.data_collection_s3fd import DataCollectionS3fd
        data_collection_pass = DataCollectionS3fd(video_data, frames, device=args.device)
    elif args.detection_method == 'haar':
        from passes.data_collection_haar_cascades import DataCollectionFaceCascade
        data_collection_pass = DataCollectionFaceCascade(video_data, frames)
    elif args.detection_method == 'mtcnn':
        from passes.data_collection_mtcnn import DataCollectionMTCNN
        data_collection_pass = DataCollectionMTCNN(video_data, frames)
    elif args.detection_method == 'dfl':
        data_collection_pass = DataCollectionDeepFaceLab(video_data, frames)
    elif args.detection_method == 'yunet':
        from passes.data_collection_yunet_cuda import DataCollectionYuNet
        data_collection_pass = DataCollectionYuNet(video_data, frames, model_path=args.yunet_model, force_cuda=True, gpu_id=0)
    elif args.detection_method == 'scrfd':
        from passes.data_collection_scrfd import DataCollectionSCRFD
        data_collection_pass = DataCollectionSCRFD(
        video_data, frames,
        onnx_path=args.scrfd_onnx,
        det_size=args.scrfd_size,
        det_thresh=args.scrfd_thresh,
        providers=tuple(args.scrfd_providers) 
    )
    else:
        raise Exception(f'{args.detection_method} is invalid.')

    if args.interpolation:
        return VideoProcessor([
            data_collection_pass,
            NearestNeighborPass(video_data, frames, search_depth=args.interpolation),
            FaceInterpPass(video_data, frames),
            RemoveNeighborsPass(video_data, frames),
            PickleEncodePass(video_data, frames, path=video_data.out_path),
            StatisticsPass(video_data, frames),
        ])
    else:
        return VideoProcessor([
            data_collection_pass,
            PickleEncodePass(video_data, frames, path=video_data.out_path),
            StatisticsPass(video_data, frames),
        ])


if __name__ == '__main__':
    if os.path.isdir(args.input_path) and os.path.isdir(args.output_path):
        print(f'will process all files in {args.input_path} to {args.output_path}.')
        in_files = glob.glob(f'{args.input_path}//*')
        for in_file in in_files:
            if path_leaf(in_file).split(".")[1] not in ['mp4', 'avi', 'mov']:
                continue
            out_file = f'{args.output_path}//{path_leaf(in_file).split(".")[0]}.pickle'
            if args.skip_processed_files and os.path.exists(out_file):
                print("Skipping {0}, file has already been pickled.".format(in_file))
            else:
                create_processor(VideoData(in_file, out_file), []).process()

    elif os.path.isfile(args.input_path):
        print(f'will process {args.input_path} to {args.output_path}.')
        if args.skip_processed_files and os.path.exists(args.output_path):
            print("Skipping {0}, file has already been pickled.".format(args.input_path))
        else:
            create_processor(VideoData(args.input_path, args.output_path), []).process()

    else:
        raise Exception(f'Possible i/o error.  Check for directory/path mismatch between input and output arguments.')

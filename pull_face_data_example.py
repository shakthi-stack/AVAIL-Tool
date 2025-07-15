from passes.pickle_pass import PickleDecodePass
from util.objects import VideoData
from util.video_processor import VideoProcessor


if __name__ == "__main__":

    input_path, pickle_path = 'workspace/example_video.mp4', 'workspace/example_video.pickle'

    video_data = VideoData(input_path, "")
    frames = []

    processor = VideoProcessor([
        PickleDecodePass(video_data, frames, path=pickle_path),
    ])

    processor.process()

    print(f'number of frames: {len(frames)}')
    print(f'face 0 at frame 0: {frames[0].faces[0]}')

class VideoProcessor:
    def __init__(self, passes):
        self.passes = passes

    def process(self):
        video_data = None
        frames = None

        for p in self.passes:
            p.execute()
            video_data, frames = p.get_values()

        return video_data, frames

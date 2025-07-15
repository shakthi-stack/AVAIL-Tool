from passes.base_pass import *
import math


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def neighbor_check(face, neighbor, offset, search_percentage):
    face_length = (face.w + face.h) / 2
    neighbor_length = (neighbor.w + neighbor.h) / 2
    cutoff_distance = search_percentage * offset * face_length
    actual_distance = distance(face.get_position(), neighbor.get_position())
    if actual_distance <= cutoff_distance:
        # check sizes as well, this one doesn't depend on frame offset
        if face_length * (1 - search_percentage) <= neighbor_length <= face_length * (1 + search_percentage):
            return actual_distance
    return -1


class NearestNeighborPass(BasePass):
    def __init__(self, video_data, frames, search_depth=5, search_percentage=.25):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.search_depth = search_depth
        self.search_percentage = search_percentage

    def execute(self):
        super().execute()
        forward = NNForwardPass(self.video_data, self.frames, self.search_depth, self.search_percentage)
        forward.execute()
        self.video_data, self.frames = forward.get_values()

        backward = NNBackwardPass(self.video_data, self.frames, self.search_depth, self.search_percentage)
        backward.execute()
        self.video_data, self.frames = backward.get_values()

    def get_values(self):
        return self.video_data, self.frames


class NNForwardPass(BasePass):
    def __init__(self, video_data, frames, search_depth, search_percentage):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.search_depth = search_depth
        self.search_percentage = search_percentage

    def execute(self):
        super().execute()

        for frame in self.frames:
            idx = frame.index
            for face in frame.faces:
                # neighbor is represented as (<face obj>, <distance>, <frames ahead>)
                neighbor = (None, 9999, 0)
                offset = 1
                while neighbor[0] is None and offset <= self.search_depth and idx + offset < len(self.frames):
                    for nbor in self.frames[idx + offset].faces:
                        dist = neighbor_check(face, nbor, offset, self.search_percentage)
                        if dist is not -1 and dist < neighbor[1]:
                            neighbor = (nbor, dist, offset)
                    offset += 1
                if neighbor[0] is not None:
                    # print("neighbor found {}".format(neighbor))
                    face.forward_neighbor = neighbor

    def get_values(self):
        return self.video_data, self.frames


class NNBackwardPass(BasePass):
    def __init__(self, video_data, frames, search_depth, search_percentage):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.search_depth = search_depth
        self.search_percentage = search_percentage

    def execute(self):
        super().execute()

        for frame in self.frames:
            idx = frame.index
            for face in frame.faces:
                # neighbor is represented as (<face obj>, <distance>, <frames behind>)
                neighbor = (None, 9999, 0)
                offset = 1
                while neighbor[0] is None and offset <= self.search_depth and idx - offset >= 0:
                    for nbor in self.frames[idx - offset].faces:
                        dist = neighbor_check(face, nbor, offset, self.search_percentage)
                        if dist is not -1 and dist < neighbor[1]:
                            neighbor = (nbor, dist, offset)
                    offset += 1
                if neighbor[0] is not None:
                    # print("neighbor found {}".format(neighbor))
                    face.backward_neighbor = neighbor

    def get_values(self):
        return self.video_data, self.frames


# This makes the data serializable after interpolation
class RemoveNeighborsPass(BasePass):
    def __init__(self, video_data, frames):
        super().__init__()
        self.video_data = video_data
        self.frames = frames

    def execute(self):
        super().execute()

        for frame in self.frames:
            for face in frame.faces:
                face.forward_neighbor = None
                face.backward_neighbor = None

    def get_values(self):
        return self.video_data, self.frames

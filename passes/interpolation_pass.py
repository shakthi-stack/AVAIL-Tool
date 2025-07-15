from passes.base_pass import *
from util.objects import Face

from scipy import interpolate
from scipy.ndimage import percentile_filter

import math


def find_face_chain_length(face):
    i = 0
    if face.backward_neighbor is None:
        while face.forward_neighbor is not None:
            face = face.forward_neighbor[0]
            i += 1
    return i


def get_interp_position(face1, face2, percentage):
    pos1 = face1.get_position()
    pos2 = face2.get_position()
    x = pos1[0] + ((pos2[0] - pos1[0]) * percentage)
    y = pos1[1] + (pos2[1] - pos1[1]) * percentage

    return int(x), int(y)


def get_interp_dimensions(face1, face2, percentage):
    pos1 = face1.w, face1.h
    pos2 = face2.w, face2.h
    x = pos1[0] + (pos2[0] - pos1[0]) * percentage
    y = pos1[1] + (pos2[1] - pos1[1]) * percentage

    return int(x), int(y)


class InterpBasePass(BasePass):
    def __init__(self, video_data, frames):
        super().__init__()
        self.video_data = video_data
        self.frames = frames

    def get_values(self):
        return self.video_data, self.frames


class CurveFittingPass(InterpBasePass):
    def __init__(self, video_data, frames):
        super().__init__(video_data, frames)

    def execute(self):
        super().execute()
        x_data = []
        y_data = []
        # first and last pad the outsides of the data with the start and end detected values
        # this prevents the curve fitting going crazy with sparse datapoints at beginning/end
        first = True
        last = -1
        last_i = -1
        index = 0
        while index < len(self.frames):
            frame = self.frames[index]
            for face in frame.faces:
                if face.tag is None:
                    s = face.w + face.h / 2
                    if first:
                        for i in range(index):
                            x_data.append(i)
                            y_data.append(s)
                        first = False
                    x_data.append(index)
                    y_data.append(s)
                    last = s
                    last_i = index
                    break
            index += 1
        for i in range(last_i, index):
            x_data.append(i)
            y_data.append(last)

        # sliding window of 5 seconds, 80% filter
        y_interp = percentile_filter(y_data, 80, 300)
        func = interpolate.splrep(x_data, y_interp, s=int(len(x_data) / 300))
        self.video_data.func = func

        # x_eval = []
        # y_eval = []
        # for x in range(index):
        #     x_eval.append(x)
        #     #y_eval.append(func(x))
        #     y_eval.append(interpolate.splev(x, func))
        # fig, axs = plt.subplots(2)
        #
        # axs[0].plot(x_data, y_data)
        # axs[1].plot(x_eval, y_eval)
        # plt.show()


class FalsePositivePass(InterpBasePass):
    def __init__(self, video_data, frames, fp_frame_cutoff=10):
        super().__init__(video_data, frames)
        self.fp_frame_cutoff = fp_frame_cutoff

    def execute(self):
        super().execute()
        index = 0
        while index < len(self.frames):
            frame = self.frames[index]
            for face in frame.faces:
                if face.backward_neighbor is None:
                    # print("chain at {} with length {}".format(index, find_face_chain_length(face)))
                    if find_face_chain_length(face) < self.fp_frame_cutoff:
                        self.delete_face_chain(face, index)
            index += 1

    def delete_face_chain(self, face, index):
        faces = list()
        indices = list()
        faces.append(face)
        indices.append(index)
        while face.forward_neighbor is not None:
            index += face.forward_neighbor[2]
            face = face.forward_neighbor[0]
            faces.append(face)
            indices.append(index)
        for i in range(len(indices)):
            # print("removing {} at index {}".format(faces[i], indices[i]))
            if faces[i] in self.frames[indices[i]].faces:
                self.frames[indices[i]].faces.remove(faces[i])


class FaceInterpPass(InterpBasePass):
    def __init__(self, video_data, frames):
        super().__init__(video_data, frames)

    def execute(self):
        super().execute()
        index = 0
        while index < len(self.frames):
            frame = self.frames[index]
            for face in frame.faces:
                if face.forward_neighbor is not None and face.forward_neighbor[2] > 1:
                    self.fill_the_gap(index, face)
            self.frames[index] = remove_faces_too_close(self.frames[index])
            index += 1

    def fill_the_gap(self, index, face):
        next_face = face.forward_neighbor[0]
        # print("filling the gap from {} to {}".format(index, index + face.forward_neighbor[2]))
        # print("start: {}".format(face))
        # print("end: {}".format(next_face))
        for x in range(1, face.forward_neighbor[2]):
            percentage = x / face.forward_neighbor[2]

            new_pos = get_interp_position(face, next_face, percentage)
            new_size = get_interp_dimensions(face, next_face, percentage)
            new_face = Face(new_pos[0], new_pos[1], new_size[0], new_size[1])
            new_face.tag = face.tag
            # print("created: {}".format(new_face))
            self.frames[index + x].faces.append(new_face)


def remove_faces_too_close(frame):
    valid_faces = []
    for f in frame.faces:
        valid_faces.append(f)
    if len(frame.faces) < 2:
        return frame
    for x in range(len(frame.faces)):
        for y in range(len(frame.faces)):
            f1 = frame.faces[x]
            f2 = frame.faces[y]
            if x != y:
                l = (f1.w + f2.w + f1.h + f2.h) / 8
                try:  # python 3.8
                    if math.dist([f1.x, f1.y], [f2.x, f2.y]) < l:
                        if valid_faces.__contains__(f1) and valid_faces.__contains__(f2):
                            valid_faces.remove(f2)
                except:  # python <3.8
                    if math.hypot(f1.x - f2.x, f1.y - f2.y) < l:
                        if valid_faces.__contains__(f1) and valid_faces.__contains__(f2):
                            valid_faces.remove(f2)
    frame.faces = valid_faces
    return frame

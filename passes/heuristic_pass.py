import copy
import numpy as np

from passes.base_pass import *
from util.objects import Face


class HeuristicPass(BasePass):
    def __init__(self, video_data, frames, grow_rate=1.08, max_size=2, max_offset=240):
        super().__init__()
        self.video_data = video_data
        self.frames = frames
        self.grow_rate = grow_rate
        self.max_size = max_size
        self.max_offset = max_offset

    def clean_face(self, face):
        # I might have accidentally pulled the wrong direction when annotating faces, make sure they are right
        if face.w < 0:
            face.x = face.x + face.w
            face.w = -face.w
        if face.h < 0:
            face.y = face.y + face.h
            face.h = -face.h
        return face

    def execute(self):
        super().execute()
        if not hasattr(self.frames[0], 'manual_annotation') or self.frames[0].manual_annotation is None:
            print("Chain correction hasn't been run on {0}".format(self.video_data.in_path))
            return

        # clean all faces
        for frame in self.frames:
            for idx in range(0, len(frame.faces)):
                frame.faces[idx] = self.clean_face(frame.faces[idx])

            for j in range(len(frame.faces) - 1, -1, -1):
                if frame.faces[j].tag is 'heuristic':
                    frame.faces.remove(frame.faces[j])

        for frame in self.frames:
            idx = frame.index
            annotation = frame.manual_annotation
            if annotation is not None and annotation.has_face:
                detected = False
                for face in frame.faces:
                    if face.tag is None:
                        detected = True
                if not detected:
                    # find the closest REAL face and heuristicify?
                    offset = 1
                    forward_face = None
                    while idx + offset < len(self.frames) and forward_face is None:
                        for face in self.frames[idx + offset].faces:
                            if face.tag is None:
                                forward_face = copy.deepcopy(face)
                                break
                        offset += 1
                        if offset > self.max_offset:
                            break
                    # check backwards as well
                    backset = -1
                    back_face = None
                    while idx + backset >= 0 and back_face is None:
                        for face in self.frames[idx + backset].faces:
                            if face.tag is None:
                                back_face = copy.deepcopy(face)
                                break
                        backset -= 1
                        if -backset > self.max_offset:
                            break
                    # essentially cancel each other out while still letting it scale
                    if forward_face is not None and back_face is None:
                        back_face = forward_face
                        backset = offset
                    if forward_face is None and back_face is not None:
                        forward_face = back_face
                        offset = backset
                    if forward_face is None:
                        continue
                    offset = np.abs(offset)
                    backset = np.abs(backset)
                    total = offset + backset
                    fg = self.grow_rate ** offset
                    bg = self.grow_rate ** backset
                    grow = min(fg, bg)
                    grow = min(grow, self.max_size)
                    heuristic_face = Face(0, 0, 0, 0)
                    heuristic_face.tag = 'heuristic'
                    heuristic_face.x = int(forward_face.x * (backset / total) + back_face.x * (offset / total))
                    heuristic_face.y = int(forward_face.y * (backset / total) + back_face.y * (offset / total))
                    heuristic_face.w = int(forward_face.w * (backset / total) + back_face.w * (offset / total))
                    heuristic_face.h = int(forward_face.h * (backset / total) + back_face.h * (offset / total))
                    # grow by rate
                    nw = heuristic_face.w * grow
                    nh = heuristic_face.h * grow
                    heuristic_face.x = int(heuristic_face.x + (heuristic_face.w - nw) / 2)
                    heuristic_face.y = int(heuristic_face.y + (heuristic_face.h - nh) / 2)
                    heuristic_face.w = int(nw)
                    heuristic_face.h = int(nh)
                    self.frames[idx].faces.append(heuristic_face)

    def get_values(self):
        return self.video_data, self.frames
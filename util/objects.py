class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class DetectedObject:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_position(self):
        return self.x, self.y

    def get_midpoint(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def is_within(self, other):
        l1 = Vector2(self.x, self.y)
        r1 = Vector2(self.x + self.w, self.y + self.h)

        l2 = Vector2(other.x, other.y)
        r2 = Vector2(other.x + other.w, other.y + other.h)

        if l1.x >= l2.x and l1.y >= l2.y and r1.x <= r2.x and r1.y <= r2.y:
            return True
        return False

    def bind_to_frame(self, x_bound, y_bound):
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w >= x_bound:
            self.w = x_bound - self.x - 1
        if self.y + self.h >= y_bound:
            self.h = y_bound - self.y - 1


class Eye(DetectedObject):
    def __init__(self, x, y, w, h):
        super().__init__(x, y, w, h)


class Face(DetectedObject):
    eyes = []
    keypoints = None
    forward_neighbor = None
    backward_neighbor = None
    tag = None

    def __init__(self, x, y, w, h, keypoints=None):
        super().__init__(x, y, w, h)
        self.keypoints = keypoints

    def add_eye(self, eye):
        self.eyes.append(eye)

    def get_eyes(self):
        return self.eyes

    def __str__(self):
        return "face at ({}, {}) with size ({}, {})".format(self.x, self.y, self.w, self.h)


class FrameData:
    def __init__(self, index=0):
        self.index = index
        self.faces = []
        self.eyes = []
        self.manual_annotation = None

    def add_face(self, face):
        self.faces.append(face)

    def add_eye(self, eye):
        self.eyes.append(eye)

    def cull_eyes(self):
        indices = []
        idx = 0
        for eye in self.eyes:
            within_a_face = False
            for face in self.faces:
                if eye.is_within(face):
                    face.add_eye(eye)
                    within_a_face = True

            if not within_a_face:
                indices.append(idx)
            idx += 1
        for x in reversed(indices):
            self.eyes.pop(x)


class ManualAnnotation:
    def __init__(self, path):
        self.path = path
        self.start = False
        self.end = False
        self.has_face = False


class VideoData:
    width = 0
    height = 0
    fps = 0
    frame_count = 0
    func = None

    def __init__(self, in_path="", out_path=""):
        self.in_path = in_path
        self.out_path = out_path

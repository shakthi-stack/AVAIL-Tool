from passes.base_pass import *

import cv2
import numpy as np
import time
from scipy import interpolate


class RenderingPass(BasePass):
    def __init__(self, video_data, frames):
        super().__init__()
        self.video_data = video_data
        self.frames = frames

    def execute(self):
        super().execute()
        dimensions = self.video_data.width, self.video_data.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_input = cv2.VideoCapture(self.video_data.in_path)
        output = cv2.VideoWriter(self.video_data.out_path, fourcc, self.video_data.fps, dimensions)

        index = 0
        next_tick = 0
        tick_interval = 0.05
        while video_input.isOpened() and index < self.video_data.frame_count:
            _, img = video_input.read()
            frame_data = self.frames[index]
            img = self.modify_image(img, frame_data)
            output.write(img)

            # quick command line progress bar
            if float(index) / self.video_data.frame_count > next_tick:
                next_tick += tick_interval
                print("{0:2.0f}% ---- frame {1:5.0f} ---- time {2:10.4f} s".format(
                    float(index) / self.video_data.frame_count * 100,
                    index,
                    time.time() - self.start_time))
            index += 1

        cv2.destroyAllWindows()
        output.release()

    def modify_image(self, img, frame_data):
        print("RenderingPass: modify_image not implemented")

    def get_values(self):
        return self.video_data, self.frames


class MultiRenderingPass(RenderingPass):
    def __init__(self, rendering_passes, video_data, frames):
        super().__init__(video_data, frames)
        self.rendering_passes = rendering_passes

    def modify_image(self, img, frame_data):
        for r_pass in self.rendering_passes:
            img = r_pass.modify_image(img, frame_data)
        return img


class RenderingPassBoxes(RenderingPass):
    def modify_image(self, img, frame_data):
        for face in frame_data.faces:
            #BGR
            if face.tag is None:
                color = (255, 0, 0)
            elif face.tag == 'not child':
                color = (0, 0, 255)
            elif face.tag == 'heuristic':
                color = (0, 255, 0)
            else:
                color = (0, 0, 0)
            rectangle_around_detected_object(face, img, color=color, thickness=4)

        for eye in frame_data.eyes:
            rectangle_around_detected_object(eye, img, color=(0, 255, 0))
        return img


class RenderingPassBlur(RenderingPass):
    def __init__(self, video_data, frames, intensity=10):
        super().__init__(video_data, frames)
        self.intensity = intensity

    def modify_image(self, img, frame_data):
        for face in frame_data.faces:
            x = int(face.x)
            y = int(face.y)
            w = int(face.x + face.w)
            h = int(face.y + face.h)
            face_img = img[y:h, x:w]

            intensity = blur_intensity(face.w, face.h, self.intensity)

            face_img = cv2.blur(face_img, (intensity, intensity))
            # Insert this image back into the main img
            for xx in range(int(face.w)):
                for yy in range(int(face.h)):
                    try:
                        img[y + yy, x + xx] = face_img[yy, xx]
                    except:
                        fuhgettaboutit = None
        return img


class RenderingPassBlurChild(RenderingPassBlur):
    def modify_image(self, img, frame_data):
        for face in frame_data.faces:
            if face.tag == "not child":
                continue
            x = int(face.x)
            y = int(face.y)
            w = int(face.x + face.w)
            h = int(face.y + face.h)
            face_img = img[y:h, x:w]

            if self.video_data.func is None:
                # print("Uh oh, didn't run CurveFittingPass on the data")
                intensity = blur_intensity(face.w, face.h, self.intensity)
            else:
                s = interpolate.splev(frame_data.index, self.video_data.func)
                intensity = blur_intensity(s, s, self.intensity)
            # intensity = blur_intensity(face.w, face.h)
            try:
                face_img = cv2.blur(face_img, (intensity, intensity))
                # Insert this image back into the main img
                for xx in range(int(face.w)):
                    for yy in range(int(face.h)):
                        try: # this block fails on out of bounds pixels but keeps going
                            img[y + yy, x + xx] = face_img[yy, xx]
                        except:
                            it_is_okay = True
            except:
                weird = True

        return img


class RenderingPassBlurEverything(RenderingPassBlur):
    def modify_image(self, img, frame_data):
        # if self.video_data.func is None:
        #     print("Uh oh, didn't run CurveFittingPass on the data")
        # s = interpolate.splev(frame_data.index, self.video_data.func)
        if len(frame_data.faces) > 0:
            face = frame_data.faces[0]
            intensity = blur_intensity(face.w, face.h, self.intensity)
        else:
            intensity = blur_intensity(None, None, self.intensity)

        # if len(frame_data.faces) > 0:
        #     ints = 0
        #     for face in frame_data.faces:
        #         if face.tag == "not child":
        #             continue
        #
        #         ints += blur_intensity(face.w, face.h)
        #     ints /= len(frame_data.faces)
        #     intensity = int((self.intensity + ints) / 2)
        #     intensity = max(intensity, self.intensity)
        # else:
        #     intensity = self.intensity

        img = cv2.blur(img, (intensity, intensity))
        return img


class RenderingPassKeypoints(RenderingPass):
    def modify_image(self, img, frame_data):
        for face in frame_data.faces:
            kp = face.keypoints
            cv2.circle(img, kp["left_eye"], 3, (255, 255, 255))
            cv2.circle(img, kp["right_eye"], 3, (255, 255, 255))
            cv2.circle(img, kp["nose"], 3, (255, 255, 255))
            cv2.circle(img, kp["mouth_left"], 3, (255, 255, 255))
            cv2.circle(img, kp["mouth_right"], 3, (255, 255, 255))

        return img


class RenderingPassNearestNeighbor(RenderingPass):
    def modify_image(self, img, frame_data):
        for face in frame_data.faces:
            if face.forward_neighbor is not None:
                cv2.line(img, face.get_position(), face.forward_neighbor[0].get_position(), (0, 0, 255), 2)
                rectangle_around_detected_object(face.forward_neighbor[0], img, color=(0, 0, 255), thickness=1)

            if face.backward_neighbor is not None:
                cv2.line(img, face.get_position(), face.backward_neighbor[0].get_position(), (0, 255, 0), 2)
                rectangle_around_detected_object(face.backward_neighbor[0], img, color=(0, 255, 0), thickness=1)
        return img


def blur_intensity(w, h, intensity=10):
    # sqrt(s)
    #return int(np.sqrt((w + h) / 2))
    # linear (s) / 10
    try:
        # blur/10
        return int(np.ceil((w + h) / intensity))
        # blur/20
        #return int(np.ceil((w + h) / 10))
        # blur/30
        # return int(np.ceil((w + h) / 15))
        # blur/40
        # return int(np.ceil((w + h) / 20))
    except Exception as e:
        # either w or h was Nan
        return intensity

    # square (s)^2 / 10
    #return int(np.ceil(np.square((w + h) / 20)))


def rectangle_around_detected_object(obj, img, color=(0, 0, 0), thickness=2):
    x = int(obj.x)
    y = int(obj.y)
    w = int(obj.x + obj.w)
    h = int(obj.y + obj.h)
    cv2.rectangle(img, (x, y), (w, h), color, thickness)


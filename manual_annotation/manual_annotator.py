import sys
import pickle

from manual_annotation.annotation_utils import *

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class App(QWidget):
    def __init__(self, path, dir="frames/", scale=1):
        super().__init__()
        self.path = path
        self.dir = dir
        pre = path.split('.')[0]
        self.pickled_path = pre + "_manual_annotations.pickle"
        self.index = 0
        self.annotations = []
        self.image = QLabel(self)
        self.frame_status = QLabel(self)
        self.cmd = QLabel(self)

        self.init_data(scale)
        self.init_UI()

    def init_data(self, scale):
        if os.path.exists(self.pickled_path):
            infile = open(self.pickled_path, "rb")
            self.annotations = pickle.load(infile)
            infile.close()
            self.log("loaded pickled data from {0}".format(self.pickled_path))
        else:
            self.annotations = populate_annotations(self.path, dir=self.dir)
            self.log("loaded data from {0}".format(self.path))
            create_dir(self.path, self.annotations, dir=self.dir, scale=scale)

    def save_data(self):
        path = self.pickled_path
        outfile = open(path, "wb")
        pickle.dump(self.annotations, outfile)
        outfile.close()
        self.log("saved data to {0}\n{1}".format(path, self.print_info()))

    def init_UI(self):
        self.setWindowTitle('Manual Annotator: Phase 1 - Mark Ground Truth Regions')
        self.setGeometry(80, 80, 1, 1)
        # Top Bar Instructions
        instructions = QLabel(self)
        instructions.setText(
            "Move (<- ->); Move x10 (, .); Skip to Keyframe (k l); Start (S); End (E); Save (Enter); Print Results (Spacebar)")
        # Image
        pixmap = QPixmap(self.annotations[self.index].path)
        self.image.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addWidget(self.image)
        vbox.addWidget(self.frame_status)
        vbox.addWidget(self.cmd)
        self.setLayout(vbox)

        self.update_image()
        self.update_frame_status(self.frame_text())

        self.show()

    def keyPressEvent(self, e):
        # clear the log
        self.log("")
        # movement
        if e.key() == Qt.Key_Right:
            self.index += 1
        if e.key() == Qt.Key_Left:
            self.index -= 1
        if e.key() == Qt.Key_Period:
            self.index += 10
        if e.key() == Qt.Key_Comma:
            self.index -= 10
        if e.key() == Qt.Key_K:
            self.index = self.next_key_frame(False)
        if e.key() == Qt.Key_L:
            self.index = self.next_key_frame(True)

        # index verification
        self.index = max(0, min(self.index, len(self.annotations) - 1))
        self.update_image()

        # annotation
        if e.key() == Qt.Key_S:
            self.annotations[self.index].start = not self.annotations[self.index].start
        if e.key() == Qt.Key_E:
            self.annotations[self.index].end = not self.annotations[self.index].end
        self.annotations = set_annotations(self.annotations)
        self.update_frame_status(self.frame_text())

        # saving
        if e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
            self.save_data()

        # printing
        if e.key() == Qt.Key_Space:
            self.log(self.print_info())

    def next_key_frame(self, right):
        if right:
            step = 1
        else:
            step = -1
        index = self.index + step
        while (True):
            try:
                frame = self.annotations[index]
                if frame.start or frame.end:
                    return index
                index += step
            except:
                # we didn't find a frame, just don't move
                self.log("No keyframes to move to")
                return self.index

    def update_image(self):
        pixmap = QPixmap(self.annotations[self.index].path)
        # pixmap_resized = pixmap.scaled(1280, 1920, Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)

    def frame_text(self):
        frame = self.annotations[self.index]
        text = "Index: {0}, hasFace: {1}, Start: {2}, End: {3}".format(self.index, frame.has_face, frame.start,
                                                                       frame.end)
        return text

    def update_frame_status(self, text):
        self.frame_status.setText(text)
        if self.annotations[self.index].has_face:
            self.frame_status.setStyleSheet("background-color: lightgreen")
        else:
            self.frame_status.setStyleSheet("background-color: red")

    def log(self, text):
        self.cmd.setText(text)

    def print_info(self):
        face_count = 0
        frame_count = 0
        for frame in self.annotations:
            if frame.has_face:
                face_count += 1
            frame_count += 1
        percentage = 100 * face_count / frame_count
        text = "Frames: {0}\nFaces: {1}\nPercentage: {2}".format(frame_count, face_count, percentage)
        print(text)
        return text


def run_manual_annotator(path, dir="", scale=1):
    app = QApplication(sys.argv)
    ex = App(path, dir, scale)

    app.exec_()


if __name__ == '__main__':
    path = sys.argv[1]
    run_manual_annotator(path)

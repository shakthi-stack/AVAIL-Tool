import sys
import pickle


from util.objects import FrameData, Face
from manual_annotation.annotation_utils import *
from passes.nearest_neighbor_pass import NearestNeighborPass, RemoveNeighborsPass

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen, QFont
from PyQt5.QtCore import Qt


class App(QWidget):
    def __init__(self, video_path, pickle_path, scale=1):
        super().__init__()
        self.video_path = video_path
        pre = video_path.split('.')[0]
        self.annotation_path = pre + "_manual_annotations.pickle"
        self.pickle_path = pickle_path
        self.scale = scale

        self.index = 0
        self.frame_data = []
        self.manual_annotation_status = QLabel(self)
        self.image = QLabel(self)
        self.frame_status = QLabel(self)
        self.cmd = QLabel(self)

        self.face_begin = None
        self.face_end = None

        self.init_data()
        self.init_UI()

    def init_data(self):
        if os.path.exists(self.annotation_path):
            infile = open(self.annotation_path, "rb")
            annotations = pickle.load(infile)
            infile.close()
            self.log("loaded pickled data from {0}".format(self.annotation_path))
        else:
            print("No manual annotation data in the expected path: {0}".format(self.annotation_path))
            print("Shutting down")
            self.close()
            return
        if os.path.exists(self.pickle_path):
            infile = open(self.pickle_path, "rb")
            self.frame_data = pickle.load(infile)
            NN = NearestNeighborPass(None, self.frame_data, 1, 1)
            NN.execute()
            _, self.frame_data = NN.get_values()
            infile.close()
        else:
            print("No frame data in the expected path: {0}".format(self.pickle_path))
            print("Shutting down")
            self.close()
            return
        print("annotations {0}".format(len(annotations)))
        print("frames {0}".format(len(self.frame_data)))
        for i in range(len(annotations)):
            if i < len(self.frame_data):
                self.frame_data[i].manual_annotation = annotations[i]
                for j in range(len(self.frame_data[i].faces) - 1, -1, -1):
                    if self.frame_data[i].faces[j].tag is 'heuristic':
                        self.frame_data[i].faces.remove(self.frame_data[i].faces[j])

        for i in range(len(self.frame_data)):
            if not hasattr(self.frame_data[i], 'manual_annotation'):
                self.frame_data[i].manual_annotation = annotations[-1]

    def save_data(self):
        path = self.pickle_path
        outfile = open(path, "wb")
        RNP = RemoveNeighborsPass(None, self.frame_data)
        RNP.execute()
        _, data = RNP.get_values()
        pickle.dump(data, outfile)
        outfile.close()
        self.log("saved data to {0}\n{1}".format(path, self.print_info()))

    def init_UI(self):
        self.setWindowTitle('Manual Annotator: Phase 2 - Annotate Face Chains')
        self.setGeometry(80, 80, 1, 1)
        # Top Bar Instructions
        instructions = QLabel(self)
        instructions.setText(
            "Move (<- ->); Move x10 (, .); Skip to Keyframe (k l); Flag Face Chain (0-9)")
        # Image
        self.pixmap = QPixmap(self.frame_data[self.index].manual_annotation.path)
        self.image.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())
        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addWidget(self.manual_annotation_status)
        vbox.addWidget(self.image)
        vbox.addWidget(self.frame_status)
        vbox.addWidget(self.cmd)
        self.setLayout(vbox)

        self.update_image()
        self.update_manual_annotation_status()
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
        self.index = max(0, min(self.index, len(self.frame_data) - 1))

        # flag face chain
        if e.key() == Qt.Key_0:
            self.flag_face_chain(0)
        if e.key() == Qt.Key_1:
            self.flag_face_chain(1)
        if e.key() == Qt.Key_2:
            self.flag_face_chain(2)
        if e.key() == Qt.Key_3:
            self.flag_face_chain(3)
        if e.key() == Qt.Key_4:
            self.flag_face_chain(4)
        if e.key() == Qt.Key_5:
            self.flag_face_chain(5)
        if e.key() == Qt.Key_6:
            self.flag_face_chain(6)
        if e.key() == Qt.Key_7:
            self.flag_face_chain(7)
        if e.key() == Qt.Key_8:
            self.flag_face_chain(8)
        if e.key() == Qt.Key_9:
            self.flag_face_chain(9)

        # draw a box mode
        if e.key() == Qt.Key_Tab:
            self.add_drawn_face()

        # update everything
        self.update_image()
        self.update_manual_annotation_status()
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
                frame = self.frame_data[index].manual_annotation
                if frame.start or frame.end:
                    return index
                for face in self.frame_data[index].faces:
                    if face.backward_neighbor is None or face.forward_neighbor is None:
                        return index
                index += step
            except:
                # we didn't find a frame, just don't move
                self.log("No keyframes to move to")
                return self.index

    def update_image(self):
        pixmap = QPixmap(self.frame_data[self.index].manual_annotation.path)
        # self.pixmap = pixmap.scaled(1920, 1080, Qt.KeepAspectRatio)
        self.pixmap = pixmap
        self.draw_face_boxes()
        self.image.setPixmap(self.pixmap)

    def add_drawn_face(self):
        x1 = int((self.face_begin.x() - 12))
        y1 = int((self.face_begin.y() - 50))
        x2 = int((self.face_end.x() - 12))
        y2 = int((self.face_end.y() - 50))
        self.log("({} {}) ({} {})".format(x1, y1, x2, y2))
        self.add_face(x1, y1, x2, y2)

    def add_face(self, x1, y1, x2, y2):
        x1 /= self.scale
        x2 /= self.scale
        y1 /= self.scale
        y2 /= self.scale
        face = Face(x1, y1, x2 - x1, y2 - y1)
        self.frame_data[self.index].faces.append(face)

    def draw_face_boxes(self):
        # create painter instance with pixmap
        painterInstance = QPainter(self.pixmap)
        # larger font
        font = QFont()
        font.setPointSize(font.pointSize() * 3)
        painterInstance.setFont(font)

        frame = self.frame_data[self.index]
        faces = frame.faces
        idx = 0
        for face in faces:
            # set rectangle color and thickness
            if face.tag is None:
                penRectangle = QPen(Qt.blue)
            else:
                penRectangle = QPen(Qt.red)
            penRectangle.setWidth(3)
            painterInstance.setPen(penRectangle)

            x = face.x * self.scale
            y = face.y * self.scale
            h = face.h * self.scale
            w = face.w * self.scale

            painterInstance.drawText(x, y+h, str(idx))
            painterInstance.drawRect(x, y, w, h)
            idx += 1
        painterInstance.end()

    def frame_text(self):
        frame = self.frame_data[self.index]
        text = ""
        for face in frame.faces:
            t = "({} {}) ({} {})".format(face.x, face.y, face.x + face.w, face.y + face.h)
            text += t
        text = "Index: {0}, faces: {1}".format(self.index, text)
        return text

    def update_manual_annotation_status(self):
        if self.frame_data[self.index].manual_annotation.has_face:
            self.manual_annotation_status.setStyleSheet("background-color: lightgreen")
            if self.frame_data[self.index].manual_annotation.start or self.frame_data[self.index].manual_annotation.end:
                self.manual_annotation_status.setStyleSheet("background-color: green")
        else:
            self.manual_annotation_status.setStyleSheet("background-color: red")

    def update_frame_status(self, text):
        self.frame_status.setText(text)

    def flag_face_chain(self, idx):
        frame = self.frame_data[self.index]
        try:
            face = frame.faces[idx]
            # flag the chain
            tag = None
            if face.tag is None:
                tag = "not child"

            org_face = frame.faces[idx]
            i = self.index
            while face.backward_neighbor is not None:
                face = face.backward_neighbor[0]
                i -= 1
            while face.forward_neighbor is not None:
                face.tag = tag
                face = face.forward_neighbor[0]
                i += 1
                if i == self.index:
                    face = org_face
            face.tag = tag
        except:
            self.log("No face at selected index")
        return

    def log(self, text):
        self.cmd.setText(text)

    def print_info(self):
        face_count = 0
        frame_count = 0
        for frame in self.frame_data:
            try:
                if frame.manual_annotation.has_face:
                    for face in frame.faces:
                        if face.tag is None:
                            face_count += 1
                    frame_count += 1
            except Exception as e:
                print(str(e))
        percentage = 100 * face_count / frame_count
        text = "Ground Truth Num Frames: {0}\nDetected Faces Within Range: {1}\nPercentage: {2}".format(frame_count, face_count, percentage)
        print(text)
        return text

    def mousePressEvent(self, event):
        self.face_begin = event.pos()
        self.face_end = event.pos()
        self.log("{} | {}".format(self.face_begin, self.face_end))

    def mouseMoveEvent(self, event):
        self.face_end = event.pos()
        self.log("{} | {}".format(self.face_begin, self.face_end))

    def mouseReleaseEvent(self, event):
        #self.face_begin = event.pos()
        self.face_end = event.pos()
        self.log("{} | {}".format(self.face_begin, self.face_end))


def run_chain_corrections(video_path, pickle_path, scale=1):
    app = QApplication(sys.argv)
    ex = App(video_path, pickle_path, scale)

    app.exec_()


if __name__ == '__main__':
    video_path = sys.argv[1]
    pickle_path = sys.argv[2]
    run_chain_corrections(video_path, pickle_path)

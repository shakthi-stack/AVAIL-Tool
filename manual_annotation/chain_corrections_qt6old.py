import os 
import sys
import pickle
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]            # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.objects import FrameData, Face
from manual_annotation.annotation_utils import *
from passes.nearest_neighbor_pass import NearestNeighborPass, RemoveNeighborsPass

from PyQt6.QtGui import QPixmap, QPen
from PyQt6.QtWidgets import (
     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
     QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QLabel
 )
from PyQt6.QtCore import Qt, QSize, QTimer

def _excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(msg, flush=True)
    try:
        # if you have a logView, dump it there too
        from PyQt6.QtWidgets import QApplication
        w = QApplication.instance().activeWindow()
        if hasattr(w, 'logView'):
            w.logView.append(msg)
    except:
        pass

sys.excepthook = _excepthook

class FaceRect(QGraphicsRectItem):
    def __init__(self, face, *args):
        super().__init__(*args)
        self.face = face

    def itemChange(self, change, value):
        if self.scene() is None:
            return super().itemChange(change, value)
        parent = self.scene().views()[0].parent()
        w_ratio, h_ratio = parent._w_ratio or 1.0, parent._h_ratio or 1.0
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.face.x = value.x() / w_ratio
            self.face.y = value.y() / h_ratio
        elif change == QGraphicsItem.GraphicsItemChange.ItemTransformChange:
            r = self.rect()
            self.face.w = r.width()  / w_ratio
            self.face.h = r.height() / h_ratio
        return super().itemChange(change, value)

class AnnotView(QGraphicsView):
    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        rb = self.rubberBandRect()
        if rb.isNull():
            return
        parent = self.parent()
        scene_rect = self.mapToScene(rb).boundingRect()
        x = scene_rect.x() / parent._w_ratio
        y = scene_rect.y() / parent._h_ratio
        w = scene_rect.width()  / parent._w_ratio
        h = scene_rect.height() / parent._h_ratio
        f = Face(x, y, w, h)
        parent.frame_data[parent.index].faces.append(f)
        parent._draw_rects()
        self.viewport().update()
    
class App(QWidget):
    def __init__(self, video_path, pickle_path, scale=1):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.setInterval(40)      #25 FPS
        self.timer.timeout.connect(self._next_frame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.video_path = video_path
        pre = video_path.split('.')[0]
        self.annotation_path = pre + "_manual_annotations.pickle"
        self.pickle_path = pickle_path
        self.scale = scale

        self.index = 0
        self.frame_data = []
        self.manual_annotation_status = QLabel(self)
        self.frame_status = QLabel(self)
        self.cmd = QLabel(self)

        # self.face_begin = None
        # self.face_end = None

        self.init_data()
        self.init_UI()

    def init_data(self):
            # 1. Load frame data first
        if not os.path.exists(self.pickle_path):
            print(f"No frame data at {self.pickle_path}; shutting down"); self.close(); return
        with open(self.pickle_path, "rb") as f:
            self.frame_data = pickle.load(f)
        NN = NearestNeighborPass(None, self.frame_data, 1, 1)
        NN.execute(); _, self.frame_data = NN.get_values()

        # 2. Load or create annotations
        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, "rb") as f:
                annotations = pickle.load(f)
            self.log(f"loaded pickled data from {self.annotation_path}")
        else:
                # ── create a dummy annotation list that points to real JPEGs ──
            from manual_annotation.annotation_utils import ManualAnnotation

            frames_root = ROOT / "workspace" / "frames"
            video_stem  = Path(self.video_path).stem          # e.g. "example_video"

            def jpg_path(idx: int) -> str:
                nested = frames_root / video_stem / f"{idx}.jpg"
                if nested.exists():
                    return str(nested)
                flat = frames_root / f"{idx}.jpg"
                return str(flat) if flat.exists() else ""

            annotations = [ManualAnnotation(jpg_path(fd.index))
                        for fd in self.frame_data]

            print("No manual annotation file found – using dummy list.")

        # if os.path.exists(self.pickle_path):
        #     infile = open(self.pickle_path, "rb")
        #     self.frame_data = pickle.load(infile)
        #     NN = NearestNeighborPass(None, self.frame_data, 1, 1)
        #     NN.execute()
        #     _, self.frame_data = NN.get_values()
        #     infile.close()
        # else:
        #     print("No frame data in the expected path: {0}".format(self.pickle_path))
        #     print("Shutting down")
        #     self.close()
        #     return
        print("annotations {0}".format(len(annotations)))
        print("frames {0}".format(len(self.frame_data)))
        for i in range(len(annotations)):
            if i < len(self.frame_data):
                self.frame_data[i].manual_annotation = annotations[i]
                for j in range(len(self.frame_data[i].faces) - 1, -1, -1):
                    if self.frame_data[i].faces[j].tag == 'heuristic':
                        self.frame_data[i].faces.remove(self.frame_data[i].faces[j])

        for i in range(len(self.frame_data)):
            if not hasattr(self.frame_data[i], 'manual_annotation'):
                self.frame_data[i].manual_annotation = annotations[-1]
                
    def _clear_scene(self):
        self.scene.clear()
        if hasattr(self, 'pix_item'):
            self.scene.addItem(self.pix_item)
            self.pix_item.setZValue(0)
        self._rect_items = []

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
        toolbar = QHBoxLayout()                              
        btnPrev  = QPushButton("Previous Frame")
        btnPlay  = QPushButton("Play")
        btnPause = QPushButton("Pause")
        btnNext  = QPushButton("Next Frame")

        btnPrev.clicked.connect(self._prev_frame)
        btnNext.clicked.connect(self._next_frame)
        btnPlay.clicked.connect(lambda: self.timer.start())
        btnPause.clicked.connect(lambda: self.timer.stop())

        toolbar.addWidget(btnPrev); toolbar.addWidget(btnPlay)
        toolbar.addWidget(btnPause); toolbar.addWidget(btnNext)

        # 0. graphics scene
        self.scene = QGraphicsScene(self)
        # self.view  = QGraphicsView(self.scene, self)
        self.view = AnnotView(self.scene, self)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addWidget(self.manual_annotation_status)
        vbox.addWidget(self.view)
        vbox.addLayout(toolbar)
        vbox.addWidget(self.frame_status)
        vbox.addWidget(self.cmd)
        self.setLayout(vbox)

        self.update_image()
        self.update_manual_annotation_status()
        self.update_frame_status(self.frame_text())

        self.show()
        self.setFocus()

    def keyPressEvent(self, e):
        # clear the log
        self.log("")
        # movement
        if e.key() == Qt.Key.Key_Right:
            self.index += 1
        elif e.key() == Qt.Key.Key_Left:
            self.index -= 1
        elif e.key() == Qt.Key.Key_Period:
            self.index += 10
        elif e.key() == Qt.Key.Key_Comma:
            self.index -= 10
        elif e.key() == Qt.Key.Key_K:
            self.index = self.next_key_frame(False)
        elif e.key() == Qt.Key.Key_L:
            self.index = self.next_key_frame(True)
        # index verification
        self.index = max(0, min(self.index, len(self.frame_data) - 1))

        # flag face chain
        if e.key() == Qt.Key.Key_0:
            self.flag_face_chain(0)
        elif e.key() == Qt.Key.Key_1:
            self.flag_face_chain(1)
        elif e.key() == Qt.Key.Key_2:
            self.flag_face_chain(2)
        elif e.key() == Qt.Key.Key_3:
            self.flag_face_chain(3)
        elif e.key() == Qt.Key.Key_4:
            self.flag_face_chain(4)
        elif e.key() == Qt.Key.Key_5:
            self.flag_face_chain(5)
        elif e.key() == Qt.Key.Key_6:
            self.flag_face_chain(6)
        elif e.key() == Qt.Key.Key_7:
            self.flag_face_chain(7)
        elif e.key() == Qt.Key.Key_8:
            self.flag_face_chain(8)
        elif e.key() == Qt.Key.Key_9:
            self.flag_face_chain(9)

        # update everything
        self.update_all()
        # self.update_image()
        # self.update_manual_annotation_status()
        # self.update_frame_status(self.frame_text())

        # saving
        if e.key() == Qt.Key.Key_Enter or e.key() == Qt.Key.Key_Return:
            self.save_data()

        # printing
        if e.key() == Qt.Key.Key_Space:
            self.log(self.print_info())
    
    
    # def _next_frame(self):
    #     self.index = min(self.index + 1, len(self.frame_data) - 1)
    #     self.update_all()

    def _next_frame(self):
        n = self.index + 1
        # fast-forward past any missing frames
        print(f"[DEBUG] _next_frame starting from index={self.index}", flush=True)
        while n < len(self.frame_data):
            raw = self.frame_data[n].manual_annotation.path
            if raw and os.path.exists(raw):
                self.index = n
                break
            n += 1
        # if nothing left, clamp to last
        else:
            self.index = len(self.frame_data) - 1
        self.update_all()

    def _prev_frame(self):
        self.index = max(self.index - 1, 0)
        self.update_all()

    def update_all(self):
        try:
            self.update_image()
            self.update_manual_annotation_status()
            self.update_frame_status(self.frame_text())
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb, flush=True)
            if hasattr(self, 'logView'):
                self.logView.append(tb)

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
        print(f"[DEBUG] update_image: index={self.index}", flush=True)
        raw = self.frame_data[self.index].manual_annotation.path
        print(f"[DEBUG] raw path: {raw!r}, exists={os.path.exists(raw)}", flush=True)
        # 1) Clear out everything except pix_item placeholder
        self._clear_scene()  

        # 2) Bail if no file
        if not raw or not os.path.exists(raw):
            self.log(f"Missing frame image: {raw}")
            return

        # 3) Load & scale
        orig = QPixmap(raw)
        if orig.isNull():
            self.log(f"Cannot load image: {raw}")
            return
        target = self.view.viewport().size()
        if target.width() == 0:
            target = QSize(320,240)
        scaled = orig.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 4) Add it to the scene
        if not hasattr(self, 'pix_item'):
            self.pix_item = self.scene.addPixmap(scaled)
        else:
            self.pix_item.setPixmap(scaled)

        # 5) Reset the view
        self.view.setSceneRect(self.pix_item.boundingRect())
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        br = self.pix_item.boundingRect()
        self._orig_w = max(1, orig.width())
        self._orig_h = max(1, orig.height())
        self._w_ratio = br.width()  / self._orig_w
        self._h_ratio = br.height() / self._orig_h
        print(f"[DEBUG] ratios w={self._w_ratio:.3f} h={self._h_ratio:.3f}", flush=True)
        if not self.timer.isActive() and hasattr(self, 'pix_item'):
            self.timer.start()

        

    def _draw_rects(self):
        # clear prior rects
        if hasattr(self, '_rect_items'):
            for it in self._rect_items:
                self.scene.removeItem(it)
        self._rect_items = []

        faces = self.frame_data[self.index].faces
        if not faces:
            self._clear_scene()
            br = self.pix_item.boundingRect()
            self._w_ratio = br.width()  / max(1, self._orig_w)
            self._h_ratio = br.height() / max(1, self._orig_h)
            return
        
        for face in faces:
            # rect = QGraphicsRectItem(face.x * self._w_ratio,
            #                         face.y * self._h_ratio,
            #                         face.w * self._w_ratio,
            #                         face.h * self._h_ratio)
            rect = FaceRect(face, face.x * self._w_ratio, face.y * self._h_ratio, face.w * self._w_ratio, face.h * self._h_ratio)
            color = Qt.GlobalColor.blue if face.tag is None else Qt.GlobalColor.red
            pen   = QPen(color); pen.setWidth(2)
            rect.setPen(pen)
            rect.setZValue(1)

            # make it editable
            rect.setFlags(
                QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable |
                QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable |
                QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges
            )
            rect._face = face
            # rect.itemChange = self._rect_item_change    # hook
            self.scene.addItem(rect)
            self._rect_items.append(rect)
    
    

    

    

    
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

def run_chain_corrections(video_path, pickle_path, scale=1):
    app = QApplication(sys.argv)
    ex = App(video_path, pickle_path, scale)

    app.exec_()
def build_chain_widget(video_path: str,
                       pickle_path: str,
                    #    frames_dir: str,
                       scale: float = 1.0):
  
    return App(video_path, pickle_path, scale)  # App is the main class


if __name__ == '__main__':
    video_path = sys.argv[1]
    pickle_path = sys.argv[2]
    run_chain_corrections(video_path, pickle_path)

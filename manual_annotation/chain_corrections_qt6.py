import os
import sys
import pickle
from pathlib import Path
import cv2
from bisect import bisect_right, bisect_left

from util.objects import Face
from manual_annotation.annotation_utils import ManualAnnotation
from passes.nearest_neighbor_pass import NearestNeighborPass, RemoveNeighborsPass
from timeline import Timeline 

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QSizeF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QPixmap, QPen, QCursor, QColor, QBrush, QTransform
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem, QToolButton, QStyle, QFrame, QSpinBox, QPlainTextEdit, QMessageBox
)

def iou(a: Face, b: Face) -> float:
  
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0

# Constants
HANDLE_SIZE = 8  # px
CYAN = QColor("#00FFFF")
HALO = QColor(0, 0, 0, 128)

class FaceRect(QGraphicsRectItem):
   
    # selected = pyqtSignal(object)

    def __init__(self, face: Face, rect: QRectF, parent_widget: 'App'):
        super().__init__(rect)
        self.face = face
        self.app  = parent_widget  # back‑pointer to outer widget
        # paint
        # self.setPen(QPen(Qt.GlobalColor.blue, 2))
        # halo = QGraphicsRectItem(rect.adjusted(-1, -1, 1, 1))
        # halo.setPen(QPen(HALO, 1))
        # halo.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        # halo.setZValue(0)
        # parent_widget.scene.addItem(halo)
        # remember so we can move it together (optional)
        # self._halo_item = halo

        self.setPen(QPen(CYAN, 4))
        self.setZValue(1)
        # interactivity flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable    |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        
        self._dragging_handle = None  # "tl", "tr", "bl", "br" or None

    
    def _scene_to_face(self, r: QRectF):
        wr, hr = self.app._w_ratio, self.app._h_ratio
        self.face.x = r.x() / wr
        self.face.y = r.y() / hr
        self.face.w = r.width()  / wr
        self.face.h = r.height() / hr

  
    def hoverMoveEvent(self, event):
        pos = event.pos()
        r   = self.rect()
        handle = None
        if QRectF(r.topLeft()   - QPointF(HANDLE_SIZE, HANDLE_SIZE), QSizeF(HANDLE_SIZE*2, HANDLE_SIZE*2)).contains(pos):
            handle = "tl"; cursor = Qt.CursorShape.SizeFDiagCursor
        elif QRectF(r.topRight() - QPointF(HANDLE_SIZE, HANDLE_SIZE), QSizeF(HANDLE_SIZE*2, HANDLE_SIZE*2)).contains(pos):
            handle = "tr"; cursor = Qt.CursorShape.SizeBDiagCursor
        elif QRectF(r.bottomLeft()- QPointF(HANDLE_SIZE, HANDLE_SIZE), QSizeF(HANDLE_SIZE*2, HANDLE_SIZE*2)).contains(pos):
            handle = "bl"; cursor = Qt.CursorShape.SizeBDiagCursor
        elif QRectF(r.bottomRight()- QPointF(HANDLE_SIZE, HANDLE_SIZE), QSizeF(HANDLE_SIZE*2, HANDLE_SIZE*2)).contains(pos):
            handle = "br"; cursor = Qt.CursorShape.SizeFDiagCursor
        else:
            cursor = Qt.CursorShape.SizeAllCursor if self.isSelected() else Qt.CursorShape.ArrowCursor
        self.setCursor(QCursor(cursor))
        self._hover_handle = handle
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        self._dragging_handle = getattr(self, "_hover_handle", None)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_handle:
            r   = self.rect()
            p   = event.pos()
            if "l" in self._dragging_handle:
                r.setLeft(p.x())
            if "r" in self._dragging_handle:
                r.setRight(p.x())
            if "t" in self._dragging_handle:
                r.setTop(p.y())
            if "b" in self._dragging_handle:
                r.setBottom(p.y())
           
            if r.width() < 2:  r.setWidth(2)
            if r.height()< 2:  r.setHeight(2)
            self.setRect(r)
            self._scene_to_face(self.mapRectToScene(r))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging_handle = None
        super().mouseReleaseEvent(event)
        if self.isSelected():
            # self.selected.emit(self)
            self.app._show_chain_controls(self)

    def itemChange(self, change: 'QGraphicsItem.GraphicsItemChange', value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self._dragging_handle:
            self._scene_to_face(self.mapRectToScene(self.rect()))
        
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # self.setPen(QPen(Qt.GlobalColor.red if value else Qt.GlobalColor.blue, 2))
            self.setPen(QPen(Qt.GlobalColor.red if value else CYAN, 4))
            # if hasattr(self, "_halo_item"):
            #     self._halo_item.setRect(self.rect().adjusted(-1, -1, 1, 1))
        return super().itemChange(change, value)



class AnnotView(QGraphicsView):
    def mousePressEvent(self, ev):
        scene_pt = self.mapToScene(ev.pos())
        if self.scene().itemAt(scene_pt, QTransform()) is None:
            self.parent()._hide_chain_controls()

        super().mousePressEvent(ev)  

    def mouseReleaseEvent(self, ev):
        rb = self.rubberBandRect()
        super().mouseReleaseEvent(ev)
        if self.dragMode() != QGraphicsView.DragMode.RubberBandDrag:
            return
            
        
        if rb.isNull():
            return
        p   = self.parent()
        wr, hr = p._w_ratio, p._h_ratio
        scen = self.mapToScene(rb).boundingRect()
        f = Face(scen.x()/wr, scen.y()/hr, scen.width()/wr, scen.height()/hr)
        p.frame_data[p.index].faces.append(f)
        p._draw_rects()
        # p.propagate_face_forward(f, from_idx=p.index,K=10, force=True)
        p.markerAdded.emit(p.index, "add")
        p._show_chain_controls(p._rects[-1])

class App(QWidget):
    finished = pyqtSignal()
    positionChanged = pyqtSignal(int)
    markerAdded     = pyqtSignal(int, str)
    def __init__(self, video_path, pickle_path, scale=1):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.setInterval(40)      # ~25 fps
        self.timer.timeout.connect(self._on_tick)
        self.video_path = video_path
        pre = video_path.split('.')[0]
        self.annotation_path = pre + "_manual_annotations.pickle"
        self.pickle_path = pickle_path
        self.scale = scale

        self.index = 0
        self._w_ratio = self._h_ratio = 1.0
        self._rects   = []
        self.frame_data = []
        self.manual_annotation_status = QLabel("No Face of Interest", self)
        self.manual_annotation_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manual_annotation_status.setFixedHeight(20)
        self.manual_annotation_status.setStyleSheet("font: 10pt 'Segoe UI'; color: white;") 
        # self.image = QLabel(self)\


        self.frame_status = QPlainTextEdit(self)
        self.frame_status.setReadOnly(True)
        self.frame_status.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.frame_status.setFixedHeight(48)
        self.frame_status.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frame_status.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.frame_status.setStyleSheet(
            "background:#1e1e1e; color:#ddd; font:9pt 'Consolas'; border:1px solid #444;"
        )
        self.cmd = QPlainTextEdit(self)
        self.cmd.setReadOnly(True)
        self.cmd.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.cmd.setFixedHeight(60)
        self.cmd.setStyleSheet(
            "background:#222; color:#9f9; font:9pt 'Consolas'; border:1px solid #555;"
        )
        self.cmd.hide()
        # self.frame_status = QLabel(self)
        # self.cmd = QLabel(self)
        # self.cmd = QPlainTextEdit(self)
        # self.face_begin = None
        # self.face_end = None

        self.init_data()
        self.init_UI()

    def init_data(self):

        if not os.path.exists(self.pickle_path):
            print(f"No frame data in the expected path: {self.pickle_path}")
            print("Shutting down")
            self.close()
            return

        with open(self.pickle_path, "rb") as f:
            self.frame_data = pickle.load(f)

        NN = NearestNeighborPass(None, self.frame_data, 1, 1)
        NN.execute()
        _, self.frame_data = NN.get_values()

        if os.path.exists(self.annotation_path):
            with open(self.annotation_path, "rb") as f:
                annotations = pickle.load(f)
            self.log(f"loaded annotations from {self.annotation_path}")
        else:
            from manual_annotation.annotation_utils import ManualAnnotation
            print("No manual annotation file found – using dummy list.")

            ROOT = Path(__file__).resolve().parents[1]
            frames_root = ROOT / "workspace" / "frames"
            video_stem  = Path(self.video_path).stem

            def jpg_path(idx: int) -> str:
                nested = frames_root / video_stem / f"{idx}.jpg"
                if nested.exists():
                    return str(nested)
                flat = frames_root / f"{idx}.jpg"
                return str(flat) if flat.exists() else ""

            annotations = [
                ManualAnnotation(jpg_path(fd.index)) for fd in self.frame_data
            ]

        print(f"annotations {len(annotations)}")
        print(f"frames {len(self.frame_data)}")

        for i, fd in enumerate(self.frame_data):
            if i < len(annotations):
                fd.manual_annotation = annotations[i]
            else:                          
                fd.manual_annotation = annotations[-1]

            fd.faces = [f for f in fd.faces
                        if not (f.tag == "heuristic")]

        if not self.frame_data:
            self.log("Frame list empty — cannot continue.")
            return

    def save_data(self):
        with open(self.pickle_path, "wb") as pf:
            RNP = RemoveNeighborsPass(None, self.frame_data)
            RNP.execute(); _, data = RNP.get_values()
            pickle.dump(data, pf)

        man_list = [fd.manual_annotation for fd in self.frame_data]
        with open(self.annotation_path, "wb") as af:
            pickle.dump(man_list, af)

        self.log(f"Saved detection: >> {self.pickle_path}\n"
                f"Saved annotations: >> {self.annotation_path}\n"
                f"{self.print_info()}")
    
    def propagate_face_forward(self, face: Face, from_idx: int, K: int = 20, iou_thr: float = 0.3, force: bool = False,):
        
        for t in range(1, K + 1):
            idx = from_idx + t
            if idx >= len(self.frame_data):
                break
            fd = self.frame_data[idx]
            # does an overlapping face already exist?
            if (not force) and any(iou(face, f) >= iou_thr for f in fd.faces):
                continue
            # clone (keep same geometry)
            # clone = Face(face.x, face.y, face.w, face.h)
            # clone.cid = face.cid
            # clone.tag = face.tag
            # fd.faces.append(clone)
            fd.faces.append(Face(face.x, face.y, face.w, face.h))

    def init_UI(self):
        self.setWindowTitle('Manual Annotator: Phase 2 - Annotate Face Chains')
        self.setGeometry(80, 80, 1, 1)
        # Top Bar Instructions
        # instructions = QLabel(self)
        # instructions.setText(
        #     "Move (<- ->); Move x10 (, .); Skip to Keyframe (k l); Flag Face Chain (0-9)")


        self.scene = QGraphicsScene(self)
        self.view  = AnnotView(self.scene, self)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        self.view.setRubberBandSelectionMode(Qt.ItemSelectionMode.ContainsItemShape)

        toolbar = QHBoxLayout()

        btnPrev = QToolButton(); btnPrev.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        btnNext = QToolButton(); btnNext.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        btnPrev.clicked.connect(self._prev_frame)
        btnNext.clicked.connect(self._next_frame)

        btnPlay = QToolButton(checkable=True)
        btnPlay.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        def _toggle_play(checked: bool):
            btnPlay.setIcon(self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPause if checked else QStyle.StandardPixmap.SP_MediaPlay))
            (self.timer.start if checked else self.timer.stop)()
        btnPlay.toggled.connect(_toggle_play)

        btnBack10 = QToolButton(); btnBack10.setText("<<10")
        btnFwd10  = QToolButton(); btnFwd10.setText("10>>")
        btnBack10.clicked.connect(lambda: self._skip_frames(-10))
        btnFwd10 .clicked.connect(lambda: self._skip_frames(+10))

        btnStart = QToolButton(); btnStart.setText("Mark Start")
        btnEnd   = QToolButton(); btnEnd  .setText("Mark End")
        btnStart.setToolTip("Set or clear the first frame of a face-present span")
        btnEnd  .setToolTip("Set or clear the last frame of a face-present span")
        btnStart.clicked.connect(lambda: self._toggle_flag("start"))
        btnEnd  .clicked.connect(lambda: self._toggle_flag("end"))

        for w in (btnPrev, btnPlay, btnNext, btnBack10, btnFwd10):
            toolbar.addWidget(w)
        toolbar.addStretch(1)          
        for w in (btnStart, btnEnd):
            toolbar.addWidget(w)

        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # self.timeline = Timeline(total_frames=total, fps=fps, parent=self)
        # self.timeline = Timeline(total_frames=len(self.frame_data), parent=self)
        # self.timeline.positionChanged.connect(self._seek_to)

        
        self.chainBox = QFrame(self)                   
        self.chainBox.setFrameShape(QFrame.Shape.Panel)
       
        self.chainBox.setStyleSheet(
            "background:#333; color:white; border:1px solid #666; border-radius:4px;"
        )
        #inner
        lay = QHBoxLayout(self.chainBox)               
        lay.setContentsMargins(8, 4, 8, 0)            
        lay.setSpacing(6)
        lay.addSpacing(4)
       
        lay.addWidget(QLabel("Propagate +", self.chainBox))

       
        self.spinFrames = QSpinBox(self.chainBox)
        self.spinFrames.setRange(0, 50)
        self.spinFrames.setValue(0)
        lay.addWidget(self.spinFrames)

        lay.addWidget(QLabel("frames", self.chainBox))

        btnApply = QPushButton("Apply", self.chainBox)
        btnApply.setStyleSheet("""
            QPushButton {
                background:#28a745;        
                color:white;
                border:1px solid #4ac05a;
                border-radius:4px;
                padding:2px 10px;
                font-weight:bold;
            }
            QPushButton:hover { background:#2ecc71; }
        """)
        lay.addWidget(btnApply)

        lay.addStretch(1)

        btnDelete = QPushButton("Delete Current", self.chainBox)
        btnDelete.setStyleSheet("""
            QPushButton {
                background:#c0392b;         
                color:white;
                border:1px solid #e06755;
                border-radius:4px;
                padding:2px 10px;
                font-weight:bold;
            }
            QPushButton:hover { background:#e74c3c; }
        """)

        font = self.chainBox.font()
        font.setPointSize(9)            
        self.chainBox.setFont(font)

        self.spinFrames.setFixedWidth(60)  
        btnApply .setMinimumWidth(60)
        btnDelete.setMinimumWidth(90)
        lay.addWidget(btnDelete) 
        self.chainBox.hide()

        btnApply.clicked.connect(self._apply_propagate)
        btnDelete.clicked.connect(self._delete_current_chain)

        btnDone = QToolButton(self)
        btnDone.setText("Done")
        btnDone.clicked.connect(self._done_and_close)

        toolbar.addWidget(btnDone)       
       
        vbox = QVBoxLayout()
        # vbox.addWidget(instructions)
        vbox.addWidget(self.manual_annotation_status)
        vbox.addWidget(self.view)
        # vbox.addWidget(self.timeline)
        vbox.addLayout(toolbar)
        vbox.addWidget(self.frame_status)
        vbox.addWidget(self.cmd)
        self.setLayout(vbox)

        self.update_image()
        self.update_manual_annotation_status()
        self.update_frame_status(self.frame_text())

        self.show()
    def update_frame_status(self, text: str):
        self.frame_status.setPlainText(text)    

    def _done_and_close(self):
        self.save_data()              
        self.finished.emit()    

    def log(self, text: str):
        self.cmd.setPlainText(text)
    def _show_chain_controls(self, rect_item):
        self._active_rect = rect_item              
        self.chainBox.show()
        self.chainBox.adjustSize()

        popup_w = self.chainBox.width()
        popup_h = self.chainBox.height()
        parent_w = self.width()
        parent_h = self.height()
        self.chainBox.move(parent_w - popup_w - 16,
                        parent_h - popup_h - 16)
    
    def _hide_chain_controls(self):
        self.chainBox.hide()
        self._active_rect = None
    
    def _apply_propagate(self):
        if not self._active_rect:
            return

        total = self.spinFrames.value()   
        if total <= 1:                
            self._hide_chain_controls()
            return

        face = self._active_rect.face
        self.propagate_face_forward(face,from_idx=self.index,K=total - 1, force=True)
        self._draw_rects()
        self._hide_chain_controls()               

    def _delete_current_chain(self):
        if not self._active_rect:
            return

        face_obj = self._active_rect.face
        fd = self.frame_data[self.index]
        fd.faces = [f for f in fd.faces
                    if not (abs(f.x - face_obj.x) < 1e-4 and
                            abs(f.y - face_obj.y) < 1e-4 and
                            abs(f.w - face_obj.w) < 1e-4 and
                            abs(f.h - face_obj.h) < 1e-4)]

        self._draw_rects()
        self.markerAdded.emit(self.index, "del")
        self._hide_chain_controls()

    def _toggle_flag(self, flag: str):
        ann = self.frame_data[self.index].manual_annotation
        setattr(ann, flag, not getattr(ann, flag))        
        ann.has_face = ann.has_face or ann.start or ann.end
        self.markerAdded.emit(self.index, "flag")
        self._cached_starts = self._cached_ends = None
        self.update_manual_annotation_status()

    def _on_tick(self):
        self._next_frame()
        # self.timeline.setPosition(self.index)
        self.positionChanged.emit(self.index)
    
    def _seek_to(self, idx: int):
        self._hide_chain_controls()
        self.index = idx
        self._refresh_all()
    
    def _skip_frames(self, delta: int):
        self.index = max(0, min(self.index + delta, len(self.frame_data)-1))
        self._refresh_all()
    


    def _next_frame(self):
        total = len(self.frame_data)

        if self.index + 1 < total:         
            self.index += 1
            self._refresh_all()
            return

        self.timer.stop()
        self._refresh_all()

        choice = QMessageBox.question(
            self,
            "Playback finished",
            "You reached the final frame.\n\nStart over and review again?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if choice == QMessageBox.StandardButton.Yes:
            self.index = 0
            self._refresh_all()
            self.timer.start()
        else:
            again = QMessageBox.question(
                self, "Confirm finished",
                "Are you completely done with this video?\n"
                "Choose <Yes> to stay on the final frame.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if again == QMessageBox.StandardButton.Yes:
                self.timer.stop()          
            else:
                self.timer.start()      

        

    def _prev_frame(self):
        self._hide_chain_controls()
        if self.index > 0:
            self.index -= 1
            self._refresh_all()

    def _refresh_all(self):
        self.update_image()
        # self.timeline.setPosition(self.index)
        self.positionChanged.emit(self.index)
        self.update_manual_annotation_status()
        self.update_frame_status(self.frame_text())


    def keyPressEvent(self, e):
        self.log("")

        if e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            changed = False
            for item in list(self.scene.selectedItems()):
                if isinstance(item, FaceRect):
                    face = item.face
                    if face in self.frame_data[self.index].faces:
                        self.frame_data[self.index].faces.remove(face)
                        self.scene.removeItem(item)
                        changed = True
            if changed:
                self._draw_rects()        
                self.markerAdded.emit(self.index, "del")
            return                     
      
        if e.key() == Qt.Key.Key_Right:
            self.index += 1
        if e.key() == Qt.Key.Key_Left:
            self.index -= 1
        if e.key() == Qt.Key.Key_Period:
            self.index += 10
        if e.key() == Qt.Key.Key_Comma:
            self.index -= 10
        if e.key() == Qt.Key.Key_K:
            self.index = self.next_key_frame(False)
        if e.key() == Qt.Key.Key_L:
            self.index = self.next_key_frame(True)
    
        self.index = max(0, min(self.index, len(self.frame_data) - 1))

        if e.key() == Qt.Key.Key_0:
            self.flag_face_chain(0)
        if e.key() == Qt.Key.Key_1:
            self.flag_face_chain(1)
        if e.key() == Qt.Key.Key_2:
            self.flag_face_chain(2)
        if e.key() == Qt.Key.Key_3:
            self.flag_face_chain(3)
        if e.key() == Qt.Key.Key_4:
            self.flag_face_chain(4)
        if e.key() == Qt.Key.Key_5:
            self.flag_face_chain(5)
        if e.key() == Qt.Key.Key_6:
            self.flag_face_chain(6)
        if e.key() == Qt.Key.Key_7:
            self.flag_face_chain(7)
        if e.key() == Qt.Key.Key_8:
            self.flag_face_chain(8)
        if e.key() == Qt.Key.Key_9:
            self.flag_face_chain(9)

        self.update_image()
        self.update_manual_annotation_status()
        self.update_frame_status(self.frame_text())

        if e.key() == Qt.Key.Key_Enter or e.key() == Qt.Key.Key_Return:
            self.save_data()

        if e.key() == Qt.Key.Key_Space:
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
                self.log("No keyframes to move to")
                return self.index

    def update_image(self):
        raw = self.frame_data[self.index].manual_annotation.path
        if not raw or not os.path.exists(raw):
            self.log(f"Missing {raw}"); return

        pix = QPixmap(raw)
        if hasattr(self, 'pix_item'):
            self.pix_item.setPixmap(pix)
        else:
            self.pix_item = self.scene.addPixmap(pix)

        self.pix_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.pix_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

        self.scene.setSceneRect(self.pix_item.boundingRect())
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

        pw, ph = max(1, pix.width()), max(1, pix.height())
        br = self.pix_item.boundingRect()
        self._w_ratio = br.width()  / pw
        self._h_ratio = br.height() / ph

        self._draw_rects()
        # if not self.timer.isActive():
        #     self.timer.start()
        
    def _draw_rects(self):
        for itm in getattr(self, '_rects', []):
            self.scene.removeItem(itm)
        self._rects = []

        for face in self.frame_data[self.index].faces:
            r = face.x*self._w_ratio, face.y*self._h_ratio, \
                face.w*self._w_ratio, face.h*self._h_ratio
            rect = FaceRect(face, QRectF(*r), self)
            # rect.selected.connect(self._show_chain_controls)
            # rect.selected.emit(rect)
            self.scene.addItem(rect)
            self._rects.append(rect)

    

    def add_face(self, x1, y1, x2, y2):
        x1 /= self.scale
        x2 /= self.scale
        y1 /= self.scale
        y2 /= self.scale
        face = Face(x1, y1, x2 - x1, y2 - y1)
        self.frame_data[self.index].faces.append(face)

    def frame_text(self):
        frame = self.frame_data[self.index]
        text = ""
        for face in frame.faces:
            t = "({} {}) ({} {})".format(face.x, face.y, face.x + face.w, face.y + face.h)
            text += t
        text = "Index: {0}, faces: {1}".format(self.index, text)
        return text

    def update_manual_annotation_status(self):
    
        ann = self.frame_data[self.index].manual_annotation
        if ann.start or ann.end:
            colour = "green"

        else:
          
            if getattr(self, "_cached_starts", None) is None:
                self._cached_starts = [i for i,f in enumerate(self.frame_data)
                                    if f.manual_annotation.start]
                self._cached_ends   = [i for i,f in enumerate(self.frame_data)
                                    if f.manual_annotation.end]

            starts = self._cached_starts
            ends   = self._cached_ends

            pos_s = bisect_right(starts, self.index) - 1
            if pos_s < 0:                         # no Start to the left
                in_span = False
            else:
                last_start = starts[pos_s]

                pos_e = bisect_left(ends, last_start)
                first_end = ends[pos_e] if pos_e < len(ends) else None

                if first_end is None:            
                    in_span = True
                else:
                    in_span = self.index <= first_end

            if in_span:
                colour = "green"
            elif ann.has_face:
                colour = "lightgreen"
            else:
                colour = "red"
        # self.manual_annotation_status.setStyleSheet(f"background-color: {colour}")
        text = "Face of Interest Present" if colour != "red" else "No Face of Interest"
        fg = "black" if colour == "lightgreen" else "white"
        self.manual_annotation_status.setText(text)
        self.manual_annotation_status.setStyleSheet(f"background-color: {colour}; color: {fg}; font: 10pt 'Segoe UI';")


    def update_frame_status(self, text):
        self.frame_status.setPlainText(text)

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

    def log(self, text: str):
        if text:
            self.cmd.setPlainText(text)
            self.cmd.show()                           
        else:
            self.cmd.clear()
            self.cmd.hide()    

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
        percentage = 0 if frame_count == 0 else 100 * face_count / frame_count
        text = "Ground Truth Num Frames: {0}\nDetected Faces Within Range: {1}\nPercentage: {2}".format(frame_count, face_count, percentage)
        print(text)
        return text

def run_chain_corrections(video_path, pickle_path, scale=1):
    app = QApplication(sys.argv)
    ex = App(video_path, pickle_path, scale)

    app.exec_()

def build_chain_widget(video_path: str, pickle_path: str, scale: float = 1.0):
    """
    Called by src/main.py after the prune step.
    Returns a fully-constructed Phase-2 widget (App) so it can be
    inserted into the top-right quadrant of the landing page.
    """
    return App(video_path, pickle_path, scale)

if __name__ == '__main__':
    video_path = sys.argv[1]
    pickle_path = sys.argv[2]
    run_chain_corrections(video_path, pickle_path)

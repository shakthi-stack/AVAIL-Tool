from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget, QSizePolicy
from collections import defaultdict

class Timeline(QWidget):

    positionChanged = pyqtSignal(int)

    def __init__(self, total_frames: int, fps: float = 30.0, parent=None):
        super().__init__(parent)
        self.total_frames = max(1, total_frames)
        self.fps = fps
        self.duration = self.total_frames / self.fps
        self.idx = 0

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)
        self.setMouseTracking(True)


        self._h_margin = 20  # px
        self._v_margin = 10  # px
        self._marks = defaultdict(set)
        self._legend_h = 14 

    def addMarker(self, idx: int, kind: str):
        self._marks[kind].add(idx)
        self.update()    

    def clearMarkers(self):
        self._marks.clear()
        self.update()

    def setPosition(self, idx: int):
        self.idx = max(0, min(idx, self.total_frames - 1))
        self.update()

    def paintEvent(self, ev):
        w, h = self.width(), self.height()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        left = self._h_margin
        right = w - self._h_margin
        track_w = right - left
        # center_y = h // 2
        center_y = self._legend_h + (h - self._legend_h) // 2

        font = QFont()
        font.setPointSize(8)
        p.setFont(font)

        total_secs = int(self.duration)

        legend = [("add",  QColor( 50,180, 50), "Add"),
                  ("del",  QColor(220, 60, 60), "Del"),
                  ("flag", QColor(230,160, 40), "Flag")]

        x = left                           
        y = 1                              
        sq = 10                           

        p.setFont(QFont(p.font().family(), 8))
        for _, col, label in legend:
          
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            p.drawRect(x, y, sq, sq)

            p.setPen(Qt.GlobalColor.black)
            p.drawText(x + sq + 4, y + sq - 1, label)

            x += sq + 4 + p.fontMetrics().horizontalAdvance(label) + 12

        for t in range(0, total_secs + 1):
            ratio = t / self.duration if self.duration > 0 else 0
            x = int(left + track_w * ratio)
            if t % 60 == 0:
                p.setPen(QPen(Qt.GlobalColor.darkGray, 2, Qt.PenStyle.SolidLine))
                p.drawLine(x, center_y - 14, x, center_y + 14)
             
                m, s = divmod(t, 60)
                text = f"{m}:{s:02d}"
                fm = p.fontMetrics()
                tw = fm.horizontalAdvance(text)
            
                p.setPen(Qt.GlobalColor.black)
                p.drawText(x - tw//2, self._v_margin + fm.ascent(), text)
            elif t % 10 == 0:
                p.setPen(QPen(QColor(120, 120, 120, 200), 1, Qt.PenStyle.DashLine))
                p.drawLine(x, center_y - 10, x, center_y + 10)
            else:
                p.setPen(QPen(QColor(200, 200, 200, 120), 1, Qt.PenStyle.DotLine))
                p.drawLine(x, center_y - 6, x, center_y + 6)

        p.setPen(QPen(Qt.GlobalColor.darkGray, 3, Qt.PenStyle.SolidLine))
        p.drawLine(left, center_y, right, center_y)

        prog = (self.idx / (self.total_frames - 1)) * track_w
        p.fillRect(QRectF(left, center_y - 6, prog, 12), QColor(100, 180, 255, 80))

        nx = int(left + prog)
        p.setPen(QPen(Qt.GlobalColor.red, 4, Qt.PenStyle.SolidLine))
        # p.drawLine(nx, self._v_margin, nx, h - self._v_margin)
        p.drawLine(nx, self._legend_h + 2, nx, h - self._v_margin)

        sec = int(self.idx / self.fps)
        m, s = divmod(sec, 60)
        tip = f"{m}:{s:02d}"
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(tip)
        th = fm.height()
        tx = nx - tw//2
        bottom_pad = self._v_margin * 1.5
        ty = h - bottom_pad - th

        rect = QRectF(tx - 4, ty - 4, tw + 8, th + 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(rect, 3, 3)
   
        p.setPen(Qt.GlobalColor.white)
        # p.drawText(tx, ty + fm.ascent(), tip)
        p.drawText(QPointF(tx, ty + fm.ascent()), tip)

        mark_h = 10                
        col = {                   
            "add": QColor( 50,180, 50),
            "del": QColor(220, 60, 60),
            "flag":QColor(230,160, 40),
        }

        for kind, frames in self._marks.items():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col[kind])
            for f in frames:
                x = int(left + track_w * f / (self.total_frames-1))
                p.drawRect(QRectF(x-1, center_y+8, 3, mark_h))

    def _x_to_index(self, x: int) -> int:
        left = self._h_margin
        track_w = self.width() - 2 * self._h_margin
        ratio = max(0.0, min(1.0, (x - left) / track_w))
        return int(ratio * (self.total_frames - 1))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            idx = self._x_to_index(int(ev.position().x()))
            self.setPosition(idx)
            self.positionChanged.emit(self.idx)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton:
            idx = self._x_to_index(int(ev.position().x()))
            self.setPosition(idx)
            self.positionChanged.emit(self.idx)

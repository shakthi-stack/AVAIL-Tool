import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Set
import pickle
import collections
from collections import Counter
import cv2
import json               
from functools import partial
from typing import NamedTuple, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANUAL_DIR = ROOT_DIR / "manual_annotation"
if str(MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(MANUAL_DIR))

from gallery_worker import GalleryWorker
from passes.nearest_neighbor_pass import NearestNeighborPass
from passes.nearest_neighbor_pass import RemoveNeighborsPass
from manual_annotation.annotation_utils import ManualAnnotation
from util.objects import VideoData
from util.video_processor import VideoProcessor
from manual_annotation.chain_corrections_qt6 import build_chain_widget
from timeline import Timeline

from PyQt6.QtCore import QObject, QThread, pyqtSignal, Qt, QSize
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QPushButton, QToolBox, QLabel, QWidget, QMessageBox, QProgressBar, QTextEdit, QListWidget, QListWidgetItem, QTabWidget, QListView  
from PyQt6 import uic
from PyQt6.QtGui import QPixmap, QImage, QIcon, QBrush

INSTRUCTION_MD = """
### Workflow

1. **Import Video**  
   *Click **Import Video…** and choose one or more *.mp4* files.*  
   The detector runs automatically; progress appears in *Log*.

2. **Review Face-Chain Gallery**  
    Click a video in the list → a new tab opens.  
   **Un-tick** whole chains you don't want.  
   Click **Continue**.

3. **Phase 2 Editing**  
   *Keyboard*   Space = Play/Pause </> = Prev/Next frame  
   *Mouse*      Draw = drag Move/Resize = drag handles  
   *Delete*     Select rect → Backspace  
   *Start/End*  Buttons toggle flags on current frame

4. **Chain Controls (pop-up)**  
   *Propagate N* = clone rect to next N frames.  
   *Delete* = remove this chain on current frame.

5. **Save & Finish**  
   Press **Enter** to save manually.  
   Press **Done ✔** to save and close the tab.

6. **Playback Loop**  
   At last frame a dialog asks to review again or stop.

*Tip:* You can keep several videos open in tabs; each remembers its own
state.
"""

class CardInfo(NamedTuple):
    cid_list: List[int]   # chains in this id cluster
    keep: bool = False    # tick to keep
    is_key: bool = False  # at most one card has this True

def count_dangling_neighbors(frames) -> int:
    # Return how many forward/backward links point to a face that no longer exists in its frame.
    idx_to_faces = {
        (fi, id(f)): f
        for fi, fr in enumerate(frames)
        for f in fr.faces
    }
    dangling = 0
    for fi, fr in enumerate(frames):
        for f in fr.faces:
            for link in ("forward_neighbor", "backward_neighbor"):
                nb = getattr(f, link, None)
                if nb and nb[0] is not None:
                    if (fi + (nb[2] if "forward" in link else -nb[2]),
                        id(nb[0])) not in idx_to_faces:
                        dangling += 1
    return dangling

print("[DEBUG] main.py loaded")

#helper to assign synthetic chain IDs after NearestNeighborPass
def assign_chain_ids(frames):
    cid_counter = 0
    visited = set()                       
    index = {
        (fidx, pidx): face
        for fidx, frame in enumerate(frames)
        for pidx, face in enumerate(frame.faces)
    }

    def add_if_valid(key, stack):
        if key in index and key not in visited:
            stack.append(key)

    for key in index:
        if key in visited:
            continue
        stack = [key]
        while stack:
            k = stack.pop()
            if k in visited:
                continue
            visited.add(k)
            face = index[k]
            setattr(face, "cid", cid_counter)     # new ID
            # follow neighbors (they’re (<face>, dist, offset))
            for link in ("forward_neighbor", "backward_neighbor"):
                nb = getattr(face, link, None)
                if nb and nb[0] is not None:
                    # reconstruct key (frame_idx + offset, same face_idx)
                    fidx, pidx = k
                    offset = nb[2]
                    target = (fidx + offset if "forward" in link else fidx - offset, pidx)
                    add_if_valid(target, stack)
        cid_counter += 1
    print(f"[DEBUG] Assigned {cid_counter} synthetic chain IDs to faces.")

def write_dummy_annotations(pickle_path: Path):
    
    with open(pickle_path, "rb") as f:
        frames = pickle.load(f)
    n_frames = len(frames)

    dummy = [ManualAnnotation("",False, False, False) for _ in range(n_frames)]
    ann_path = pickle_path.with_name(pickle_path.stem + "_manual_annotations.pickle")
    with open(ann_path, "wb") as f:
        pickle.dump(dummy, f)

    print(f"[DEBUG] wrote dummy annotations → {ann_path}")
    return ann_path
class VideoTab(QWidget):
    def __init__(self, parent: "MainWindow", video_name: str):
        super().__init__()
        self.parent      = parent
        self.video_name  = video_name
        self.video_path  = parent.WORKSPACE_DIR / video_name
        self.frames_cache: list | None = None        

        # UI skeleton 
        lay               = QVBoxLayout(self)
        self.faceList     = QListWidget()            # phase 1 gallery
        self.faceList.setViewMode(QListWidget.ViewMode.IconMode)
        self.faceList.setIconSize(QSize(96, 96))
        self.faceList.setResizeMode(QListWidget.ResizeMode.Adjust)

        self.faceList.itemClicked.connect(self._card_clicked)


        
        self.continueBtn  = QPushButton("Continue")  # phase 1  phase-2
        self.phaseBox     = QWidget()                # holds phase-2 + timeline
        self.phaseLay     = QVBoxLayout(self.phaseBox)

        lay.addWidget(self.faceList)
        lay.addWidget(self.continueBtn)
        lay.addWidget(self.phaseBox)



        self.continueBtn.clicked.connect(self.on_continue)
        
        self._parking = QWidget()

        # references filled later
        self.chain_widget:  QWidget | None = None
        self.timeline:      'Timeline' | None = None

        self._load_gallery()
    
    def _card_clicked(self, item: QListWidgetItem):
        ci: CardInfo = item.data(Qt.ItemDataRole.UserRole)

        # Ctrl-click → (de)select *key* subject
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier:
            # clear any existing key card
            

            ci = ci._replace(keep=True, is_key=not ci.is_key)

        # plain click to keep / drop toggle
        else:
            ci = ci._replace(keep=not ci.keep)
            if not ci.keep:
                ci = ci._replace(is_key=False)

        # visual cue
        bg = Qt.GlobalColor.blue if ci.is_key        else \
            Qt.GlobalColor.lightGray if ci.keep     else QBrush(Qt.BrushStyle.NoBrush)
        item.setBackground(bg)
        item.setData(Qt.ItemDataRole.UserRole, ci)

    

    def _load_gallery(self) -> None:
       
        pkl = self.video_path.with_suffix(".pickle")
        self.faceList.clear()
        self.faceList.addItem("Loading…")
        self._worker_thread = QThread(self)
        self._worker        = GalleryWorker(pkl)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_gallery_ready)  
        self._worker.finished.connect(self._worker_thread.quit)

        self._worker_thread.start()

        with open(pkl, "rb") as f:
            self.frames_cache = pickle.load(f)


    def _on_gallery_ready(self, payload):
        clusters, thumbs = payload['clusters'], payload['thumbs']
        self.faceList.clear()
        for lbl, cid_list in clusters.items():
            qimg = thumbs[lbl]                         
            it = QListWidgetItem(QIcon(QPixmap.fromImage(qimg)), "")
            ci = CardInfo(cid_list, keep=False, is_key=False)
            it.setData(Qt.ItemDataRole.UserRole, ci)
            # it.setData(Qt.ItemDataRole.UserRole, cid_list)
            it.setBackground(QBrush(Qt.BrushStyle.NoBrush))
            # it.setCheckState(Qt.CheckState.Unchecked)  
            self.faceList.addItem(it)
        self.parent.logView.append(f"{self.video_name}: loaded {len(clusters)} identities for review.")

    def on_continue(self):

        if self.frames_cache is None:
            QMessageBox.information(self, "No gallery", "Load gallery first.")
            return
        vd = VideoData(str(self.video_path), "")
        VideoProcessor([NearestNeighborPass(vd, self.frames_cache)]).process()
        assign_chain_ids(self.frames_cache)
        
        # checked_cids = {
        #     cid
        #     for i in range(self.faceList.count())
        #     if self.faceList.item(i).checkState() == Qt.CheckState.Checked
        #     for cid in self.faceList.item(i).data(Qt.ItemDataRole.UserRole)
        # }
        # if not checked_cids:
        #     QMessageBox.information(self, "Nothing selected","Tick at least one face to keep.")
        #     return
        key_cids, keep_cids = set(), set()
        for i in range(self.faceList.count()):
            ci: CardInfo = self.faceList.item(i).data(Qt.ItemDataRole.UserRole)
            if ci.keep:
                keep_cids.update(ci.cid_list)
            if ci.is_key:
                key_cids.update(ci.cid_list)
        if not key_cids:
            QMessageBox.information(self, "Choose key subject", "Ctrl-click one card to mark the key subject.")
            return

        for fd in self.frames_cache:
            new_faces = []
            for f in fd.faces:
                if f.cid not in keep_cids:
                    continue
                f.tag = None if f.cid in key_cids else "not child"
                new_faces.append(f)
            fd.faces = new_faces

        self.parent.logView.append(
            f"{self.video_name}: kept {len(keep_cids)} chains "
            f"({len(key_cids)} key). Relinking & saving…")

        
        # vd = VideoData(str(self.video_path), "")
        # VideoProcessor([NearestNeighborPass(vd, self.frames_cache)]).process()

        RemoveNeighborsPass(None, self.frames_cache).execute()

        # overwrite pickle
        with open(self.video_path.with_suffix(".pickle"), "wb") as f:
            pickle.dump(self.frames_cache, f)
        
        self.faceList.hide()
        self.continueBtn.hide()

        self.parent.logView.append(f"{self.video_name}: gallery refreshed.")
        self._embed_phase2()                             # jump to Phase-2

    def _embed_phase2(self):

        # clear previous run (if user hits Continue twice)
        while self.phaseLay.count():
            w = self.phaseLay.takeAt(0).widget()
            if w: w.setParent(None)

        chain_widget = build_chain_widget(
            str(self.video_path),
            str(self.video_path.with_suffix(".pickle")),
            scale=1.0,
        )
        self.phaseLay.addWidget(chain_widget)

        cap  = cv2.VideoCapture(str(self.video_path))
        tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        # timeline = Timeline(total_frames=tot, fps=fps, parent=self)
        # self.phaseLay.addWidget(timeline)

        timeline = Timeline(total_frames=tot, fps=fps, parent=self.parent)

        self.chain_widget = chain_widget
        self.timeline     = timeline

        bot_lay = self.parent.bottomRight.layout()
        # clear any previous timeline (from another tab)
        while bot_lay.count():
            old = bot_lay.takeAt(0).widget()
            if old:
                old.setParent(None)
        bot_lay.addWidget(timeline)

        timeline.positionChanged.connect(chain_widget._seek_to)
        chain_widget.positionChanged.connect(timeline.setPosition)
        chain_widget.markerAdded.connect(timeline.addMarker)
        chain_widget.finished.connect(self._phase2_done) 
        self.parent.logView.append(
            f"{self.video_name}: Phase-2 editor + timeline loaded.")
    def _phase2_done(self):
        QMessageBox.information(self, "Finished",
                                f"{self.video_name} saved.\n"
                                "You can close the tab or switch videos.")
        idx = self.parent.videoTabs.indexOf(self)
        if idx != -1:
            self.parent.videoTabs.removeTab(idx)
            self.parent._video_tabs.pop(self.video_name, None)

class VideoWorker(QObject):
    progress = pyqtSignal(str, str)
    finished = pyqtSignal(str, bool)

    def __init__(self, workspace: Path, filename: str):
        super().__init__()
        self.workspace = workspace
        self.filename = filename
        print(f"[DEBUG] VideoWorker created for {filename}")

    def _run_and_stream(self, cmd: List[str]) -> bool:
        print(f"[DEBUG] _run_and_stream: {' '.join(cmd)}")
        self.progress.emit(self.filename, f"$ {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout
        for line in proc.stdout:
            self.progress.emit(self.filename, line.rstrip())
        proc.stdout.close()
        ret = proc.wait()
        print(f"[DEBUG] process exit code: {ret}")
        return ret == 0

    def run(self):
        print(f"[DEBUG] Worker.run() started for {self.filename}")
        self.progress.emit(self.filename, ">>> Worker started")
        video = self.workspace / self.filename
        pickle = video.with_suffix('.pickle')
        frames = self.workspace / 'frames'
        frames.mkdir(exist_ok=True)
        py = sys.executable
        ok = (self._run_and_stream([py, '-u', 'automatic_detection.py', str(video), str(pickle), '-d', 's3fd']) and self._run_and_stream([py, '-u', 'create_frames_directory.py', str(video), str(frames)]))
        print(f"[DEBUG] Worker finished for {self.filename}: {ok}")
        self.finished.emit(self.filename, ok)


class MainWindow(QMainWindow):
    WORKSPACE_DIR = Path(__file__).resolve().parent.parent / 'workspace'

    def __init__(self):
        print("[DEBUG] MainWindow::__init__ called")
        super().__init__()
        uic.loadUi('src/ui/landing.ui', self)
        print("[DEBUG] UI loaded")
        
        self._running_workers = 0
        self._videos_in_progress: set[str] = set() 
        if hasattr(self, 'bottomRight') and self.bottomRight.layout() is None:
            self.bottomRight.setLayout(QVBoxLayout())   

       
        self.videoTabs = QTabWidget(self)
        self.videoTabs.setTabsClosable(True)
        self.videoTabs.tabCloseRequested.connect(self._close_video_tab)
        self.videoTabs.currentChanged.connect(self._on_tab_switched)
        if hasattr(self, 'topRight'):
            idx = self.rightSplitter.indexOf(self.topRight)
            self.rightSplitter.insertWidget(idx, self.videoTabs)
            self.topRight.setParent(None)               
        else:
           
            self.rightSplitter.addWidget(self.videoTabs)

    
        self._video_tabs: dict[str, VideoTab] = {}
        if not hasattr(self, 'importBtn') or not hasattr(self, 'fileList'):
            print("[DEBUG] Fallback top-left widgets")
            self._create_top_left_widgets()
        if not hasattr(self, 'infoToolBox'):
            print("[DEBUG] Fallback bottom-left widgets")
            self._create_bottom_left_widgets()
        if self.bottomRight.layout() is None:          
            self.bottomRight.setLayout(QVBoxLayout())
        self.setWindowTitle('PyQt6 Landing Page')
        self.resize(900, 600)
        self.leftSplitter.setSizes([300, 250])
        self.mainSplitter.setSizes([260, 640])
        self.rightSplitter.setSizes([520, 180])
        print(f"[DEBUG] WORKSPACE_DIR={self.WORKSPACE_DIR}")
        self.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        self.importBtn.clicked.connect(self.open_file_dialog)
        print("[DEBUG] importBtn connected to open_file_dialog")
        self._threads: List[QThread] = []
        #keeping track of the current video selection
        self.current_video = None
        self.fileList.itemClicked.connect(self._on_video_selected)
        # self.continueBtn.clicked.connect(self._apply_face_selection)
        self.continueBtn.clicked.connect(self._global_continue)
        

    def _on_video_selected(self, item, _col=0):
        name = item.text(0)
        if name in self._videos_in_progress: 
            QMessageBox.information(
                self, "Please wait",
                "Automatic detection is still running.\n"
                "Wait until it finishes before opening another video.")
            return
        if name in self._video_tabs:
            idx = self.videoTabs.indexOf(self._video_tabs[name])
            self.videoTabs.setCurrentIndex(idx)
            return

        tab = VideoTab(self, name)
        self._video_tabs[name] = tab
        self.videoTabs.addTab(tab, name)
        self.videoTabs.setCurrentWidget(tab)
    
    def _global_continue(self):
        tab = self.videoTabs.currentWidget()
        if not tab:
            QMessageBox.information(self, "Pick a video first", "Select a file row.")
            return
        tab.on_continue()
    
    def _on_tab_switched(self, index: int):
        tab = self.videoTabs.widget(index)
        if not isinstance(tab, VideoTab):
            return          # no tab or not a VideoTab
        
        if tab.timeline is None:
            return
        
        bot_lay = self.bottomRight.layout()
        while bot_lay.count():
            old = bot_lay.takeAt(0).widget()
            if old:
                old.setParent(None)
        
        bot_lay.addWidget(tab.timeline)

    def _close_video_tab(self, index: int):
        tab = self.videoTabs.widget(index)
        if not tab:
            return
        key = tab.video_name
        self.videoTabs.removeTab(index)
        self._video_tabs.pop(key, None)    



    def _create_top_left_widgets(self):
        print("[DEBUG] Creating top-left widgets in code")
        vbox = QVBoxLayout(self.topLeft)
        self.importBtn = QPushButton('Import Video…', self.topLeft)
        self.fileList = QTreeWidget(self.topLeft)
        self.fileList.setHeaderLabels(['File', 'Status'])
        self.fileList.setColumnWidth(0, 180)
        vbox.addWidget(self.importBtn)
        vbox.addWidget(self.fileList)
        vbox.setStretch(1, 1)
        self.topLeft.setLayout(vbox)

    def _create_bottom_left_widgets(self):
        print("[DEBUG] Creating bottom-left widgets in code")
        vbox = QVBoxLayout(self.bottomLeft)
        self.infoToolBox = QToolBox(self.bottomLeft)
        instr_page = QWidget()
        instr_layout = QVBoxLayout(instr_page)
        # instr_layout.addWidget(QLabel('Instructions will appear here.'))
        inst = QTextEdit(INSTRUCTION_MD, self.bottomLeft)
        inst.setReadOnly(True)
        inst.setMarkdown(INSTRUCTION_MD)
        instr_layout.addWidget(inst)
        instr_layout.addStretch()
        self.infoToolBox.addItem(instr_page, 'Instructions')
        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        self.logView = QTextEdit(log_page)
        self.logView.setReadOnly(True)
        log_layout.addWidget(self.logView)
        self.infoToolBox.addItem(log_page, 'Log')
        vbox.addWidget(self.infoToolBox)
        vbox.setStretch(0, 1)
        self.bottomLeft.setLayout(vbox)
        print("[DEBUG] Bottom-left UI setup complete")

    def open_file_dialog(self):
        print("[DEBUG] open_file_dialog triggered")
        paths, _ = QFileDialog.getOpenFileNames(self, 'Select videos', str(Path.home()), 'Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)')
        print(f"[DEBUG] Files chosen: {paths}")
        if not paths:
            return
        existing = {self.fileList.topLevelItem(i).text(0) for i in range(self.fileList.topLevelItemCount())}
        for p in paths:
            src = Path(p)
            if src.name in existing:
                continue
            dst = self._unique_destination(src.name)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                QMessageBox.warning(self, 'Copy Error', f'{src.name}: {e}')
                continue
            item = QTreeWidgetItem([dst.name, 'Queued'])
            self.fileList.addTopLevelItem(item)
            print(f"[DEBUG] Queued worker for {dst.name}")
            self._start_worker(dst.name)

    def _start_worker(self, filename: str):
        print(f"[DEBUG] _start_worker called for {filename}")
        bar = QProgressBar()
        bar.setRange(0, 0)
        for i in range(self.fileList.topLevelItemCount()):
            it = self.fileList.topLevelItem(i)
            if it.text(0) == filename:
                self.fileList.setItemWidget(it, 1, bar)
                break
        self._videos_in_progress.add(filename)
        thread = QThread(self)
        worker = VideoWorker(self.WORKSPACE_DIR, filename)
        thread.worker_ref = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_output)
        worker.finished.connect(self._on_worker_done)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        if self._running_workers == 0:
            pass                           
        self._running_workers += 1
        thread.start()
        self._threads.append((thread, worker))
        print(f"[DEBUG] Thread started for {filename}")

    def _on_worker_output(self, filename: str, line: str):
        entry = f'[{filename}] {line}'
        print(f"[DEBUG] _on_worker_output: {entry}", flush=True)
        if hasattr(self, 'logView'):
            self.logView.append(entry)

    def _on_worker_done(self, filename: str, ok: bool):
        self._running_workers -= 1
        self._videos_in_progress.discard(filename)
        print(f"[DEBUG] _on_worker_done: {filename} -> {ok}")
        status = '✓ Done' if ok else '✗ Error'
        for i in range(self.fileList.topLevelItemCount()):
            it = self.fileList.topLevelItem(i)
            if it.text(0) == filename:
                self.fileList.removeItemWidget(it, 1)
                it.setText(1, status)
                break

    def _unique_destination(self, fn: str) -> Path:
        dst = self.WORKSPACE_DIR / fn
        if not dst.exists():
            return dst
        stem, suf = dst.stem, dst.suffix
        c = 1
        while True:
            cand = self.WORKSPACE_DIR / f'{stem}_{c}{suf}'
            if not cand.exists():
                return cand
            c += 1

if __name__ == '__main__':
    print("[DEBUG] Starting application")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

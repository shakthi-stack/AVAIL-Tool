from PyQt6.QtCore import QObject, pyqtSignal, QThread
import pickle, cv2, numpy as np, pathlib
from identity_cluster import cluster_identities   

class GalleryWorker(QObject):
    finished = pyqtSignal(dict)      

    def __init__(self, pkl_path: pathlib.Path):
        super().__init__()
        self.pkl_path = pkl_path

    def run(self):  
        clusters, thumbs = cluster_identities(self.pkl_path)
        # dict clusters[label] = [cid, cid, …]
        self.finished.emit({'clusters': clusters, 'thumbs': thumbs})
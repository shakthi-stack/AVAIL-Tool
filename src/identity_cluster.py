import pickle, cv2, numpy as np, pathlib
import sys
from sklearn.cluster import AgglomerativeClustering 
from PyQt6.QtGui import QImage 

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANUAL_DIR = ROOT_DIR / "manual_annotation"
if str(MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(MANUAL_DIR))



def _ensure_chain_ids(frames, pkl_path):
    first_face = frames[0].faces[0] if frames and frames[0].faces else None
    if first_face is not None and not hasattr(first_face, "cid"):

        from util.objects import VideoData, FrameData
        from passes.nearest_neighbor_pass import NearestNeighborPass
        from util.video_processor import VideoProcessor
        from main import assign_chain_ids

        vd = VideoData(str(pkl_path.with_suffix(".mp4")), str(pkl_path))
        VideoProcessor([NearestNeighborPass(vd, frames)]).process()
        assign_chain_ids(frames)

def _embed(bgr_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cv2.resize(bgr_crop, (64, 64)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    vec = np.concatenate([h, s, v])
    return cv2.normalize(vec, vec).astype("float32") 

def _rep_crop(jpg_path: str, face):
    img = cv2.imread(jpg_path)
    if img is None:
        raise FileNotFoundError(jpg_path)

    # Try common bbox field sets
    if all(hasattr(face, a) for a in ("x", "y", "w", "h")):
        x, y, w, h = map(int, (face.x, face.y, face.w, face.h))
    elif all(hasattr(face, a) for a in ("left", "top", "right", "bottom")):
        x, y = int(face.left), int(face.top)
        w, h = int(face.right - face.left), int(face.bottom - face.top)
    else:                                # fallback: x1,y1,x2,y2
        x, y = int(face.x1), int(face.y1)
        w, h = int(face.x2 - face.x1), int(face.y2 - face.y1)

    # clamp to image bounds
    x, y = max(0, x), max(0, y)
    crop = img[y : y + h, x : x + w]
    return crop

def cluster_identities(pkl_path: pathlib.Path, thumb_size=96):
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)
    
    _ensure_chain_ids(frames, pkl_path)
    # collect one sample per chain
    chains, reps, jpgs = {}, [], {}
    for fd in frames:
        for fce in fd.faces:
            chains.setdefault(fce.cid, []).append((fd.index, fce))

    for cid, samples in chains.items():
        frm_idx, fce = samples[len(samples)//2]         # middle sample
        jpg = pkl_path.with_name("frames") / pkl_path.stem / f"{frm_idx}.jpg"
        if not jpg.exists():
            jpg = pkl_path.with_name("frames") / f"{frm_idx}.jpg"
        reps.append(_embed(_rep_crop(str(jpg), fce)))
        jpgs[cid] = (str(jpg), fce)

    emb = np.stack(reps)
    lbls = AgglomerativeClustering(
              n_clusters=None, distance_threshold=0.35,
              linkage="average").fit_predict(emb)

    clusters = {}
    thumbs   = {}
    for cid, lbl in zip(chains.keys(), lbls):
        clusters.setdefault(lbl, []).append(cid)
    for lbl, cid_list in clusters.items():
        jpg, face = jpgs[cid_list[0]]
        crop = cv2.resize(_rep_crop(jpg, face), (thumb_size, thumb_size))
        qimg = QImage(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).data,
                      thumb_size, thumb_size, 3*thumb_size,
                      QImage.Format.Format_RGB888)
        thumbs[lbl] = qimg

    return clusters, thumbs

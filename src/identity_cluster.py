import pickle, cv2, numpy as np, pathlib
import sys
from sklearn.cluster import AgglomerativeClustering 
from PyQt6.QtGui import QImage 
from itertools import combinations
from scipy.spatial.distance import cosine
import csv, datetime
from pathlib import Path

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANUAL_DIR = ROOT_DIR / "manual_annotation"
if str(MANUAL_DIR) not in sys.path:
    sys.path.insert(0, str(MANUAL_DIR))

MIN_DET_CONF  = 0.40     # Face.score threshold
MIN_BBOX_AREA = 900      # 30×30 px
MIN_CHAIN_LEN = 3        # ignore 1 or 2 frame flickers
MAX_CLUSTERS  = 4000     # cap for O(N^2) clustering

def _ensure_chain_ids(frames, pkl_path):
    need_cids = False
    for fr in frames:
        for f in fr.faces:
            if not hasattr(f, "cid"):
                need_cids = True
                break
        if need_cids:
            break
    if not need_cids:         
        return
    
    from util.objects import VideoData
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

    # # clamp to image bounds
    # x, y = max(0, x), max(0, y)
    # crop = img[y : y + h, x : x + w]
    # return crop
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    return crop if crop.size else None

def cluster_identities(pkl_path: pathlib.Path, thumb_size=96):
    with open(pkl_path, "rb") as f:
        frames = pickle.load(f)
    
    _ensure_chain_ids(frames, pkl_path)
    # collect one sample per chain
    chains, reps, jpgs, good_cids = {}, [], {}, []
    bad_chains, short_chains = 0, 0

    for fd in frames:
        for fce in fd.faces:
            chains.setdefault(fce.cid, []).append((fd.index, fce))

    for cid, samples in chains.items():
        if len(samples) < MIN_CHAIN_LEN:
            short_chains += 1
            continue

        frm_idx, fce = samples[len(samples)//2]         # middle sample

        if hasattr(fce, "score") and fce.score < MIN_DET_CONF:
            bad_chains += 1
            continue
        if fce.w * fce.h < MIN_BBOX_AREA:
            bad_chains += 1
            continue

        jpg = pkl_path.with_name("frames") / pkl_path.stem / f"{frm_idx}.jpg"
        if not jpg.exists():
            jpg = pkl_path.with_name("frames") / f"{frm_idx}.jpg"
        # reps.append(_embed(_rep_crop(str(jpg), fce)))
        crop = _rep_crop(str(jpg), fce)
        if crop is None: 
            bad_chains += 1
            continue
        reps.append(_embed(crop))
        jpgs[cid] = (str(jpg), fce)
        good_cids.append(cid) 
    print(f"[identity_cluster] filtered {bad_chains + short_chains} chains "
      f"({short_chains} short, {bad_chains} low-quality); "
      f"{len(good_cids)} remain.")
    
    #logging edits made to the pkl file (short and low quality chains removed and then clustered)
    log_dir = Path(pkl_path).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / f"{pkl_path.stem}_phase1_cluster_filter.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts","clip","short_chains","low_quality_chains","kept_chains"])
        w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            pkl_path.stem, short_chains, bad_chains, len(good_cids)
        ])

    if not reps:               
        return {}, {}
    
    emb = np.stack(reps)
    lbls = AgglomerativeClustering(
              n_clusters=None, distance_threshold=0.35,
              linkage="average", compute_full_tree="auto").fit_predict(emb)
    
    parent = {cid: cid for cid in good_cids}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    rep_vec = {cid: emb[i] for i, cid in enumerate(good_cids)}
    for a, b in combinations(good_cids, 2):
        if cosine(rep_vec[a], rep_vec[b]) < 0.25: 
            union(a, b)
    
    clusters = {}
    for cid, lbl in zip(good_cids, lbls):
        lbl = root(cid)              
        clusters.setdefault(lbl, []).append(cid)
    
    for cid_list in clusters.values():
        key = next((c for c in cid_list if getattr(jpgs[c][1], 'is_key', False)), None)
        if key is None:
            key = max(cid_list, key=lambda c: len(chains[c]))
        for c in cid_list:
            setattr(jpgs[c][1], 'is_key', c == key)
    

    clusters = {}
    thumbs   = {}
    for cid, lbl in zip(good_cids, lbls):
        clusters.setdefault(lbl, []).append(cid)
    for lbl, cid_list in clusters.items():
        # jpg, face = jpgs[cid_list[0]]
        # crop = cv2.resize(_rep_crop(jpg, face), (thumb_size, thumb_size))
        for cid in cid_list:
            if cid in jpgs:
                jpg, face = jpgs[cid]
                crop = cv2.resize(_rep_crop(jpg, face), (thumb_size, thumb_size))
                break
            else:
                continue

        qimg = QImage(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).data,
                      thumb_size, thumb_size, 3*thumb_size,
                      QImage.Format.Format_RGB888)
        thumbs[lbl] = qimg

    return clusters, thumbs

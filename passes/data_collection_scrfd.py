from util.objects import Face, FrameData
from passes.data_collection_pass import DataCollectionPass

import os, time, inspect
import numpy as np
import onnxruntime as ort
from insightface.model_zoo.scrfd import SCRFD


class DataCollectionSCRFD(DataCollectionPass):
    """
    SCRFD via ONNX Runtime using a local .onnx (no downloads).
    - Uses CUDAExecutionProvider if available, else CPU.
    - det_size must be divisible by 32 (e.g., 640x640, 736x736, 896x896).
    - Robust to InsightFace SCRFD.detect() signature differences.
    """
    def __init__(self, video_data, frames,
                 onnx_path="models/scrfd_10g_bnkps.onnx",
                 det_size=(640, 640), det_thresh=0.5,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider")
                 #providers=("CPUExecutionProvider",)
                 ):
        super().__init__(video_data, frames)

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"[SCRFD] ONNX not found: {onnx_path}")

        w, h = int(det_size[0]), int(det_size[1])
        if (w % 32) or (h % 32):
            raise ValueError(f"[SCRFD] det_size must be divisible by 32, got {det_size}")

        self.onnx_path   = onnx_path
        self.det_size    = (w, h)
        self.det_thresh  = float(det_thresh)
        self.providers   = list(providers)

        # specify threads to avoid affinity warnings/oversubscription
        so = ort.SessionOptions()
        cores = max(1, (os.cpu_count() or 2) - 1)
        so.intra_op_num_threads = cores
        so.inter_op_num_threads = 1
        # reduce CPU spin
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.session = ort.InferenceSession(self.onnx_path, sess_options=so, providers=self.providers)
        self.scrfd   = SCRFD(model_file=self.onnx_path, session=self.session)

        # Decide detect() call style once (handles InsightFace version differences)
        self._detect_mode = self._pick_detect_mode()

        self._next_tick, self._tick = 0.0, 0.05
        print(f"[SCRFD] model={os.path.basename(self.onnx_path)} det_size={self.det_size} "
              f"providers={self.session.get_providers()} threads={{intra:{cores}, inter:1}} mode={self._detect_mode}")

    # Figure out how to call SCRFD.detect in this installation
    def _pick_detect_mode(self):
        try:
            sig = inspect.signature(self.scrfd.detect)
            params = list(sig.parameters.keys()) 
            has_input_kw = 'input_size' in params
            has_thresh_kw = 'thresh' in params or 'threshold' in params or 'score_thr' in params
            if has_input_kw and has_thresh_kw:
                return "kw_input_thresh"         # detect(img, input_size=..., thresh=...)
            if has_input_kw:
                return "kw_input_only"           # detect(img, input_size=...)  
            # fallback to positional variants
            return "positional"
        except Exception:
            return "positional"

    def _call_detect(self, img):
        # Try according to detected mode; fall back through the common variants
        if self._detect_mode == "kw_input_thresh":
            try:
                return self.scrfd.detect(img, input_size=self.det_size, thresh=self.det_thresh)
            except TypeError:
                pass
        if self._detect_mode == "kw_input_only":
            try:
                return self.scrfd.detect(img, input_size=self.det_size)
            except TypeError:
                pass
        
        try:
            return self.scrfd.detect(img, self.det_size)
        except TypeError:
            pass
        
        try:
            return self.scrfd.detect(img, self.det_thresh, self.det_size)
        except TypeError:
            pass
        
        try:
            return self.scrfd.detect(img, self.det_size, self.det_thresh)
        except TypeError as e:
            raise RuntimeError(f"[SCRFD] detect() calling failed across all known signatures: {e}")

    def execute(self):
        super().execute()
        idx = 0
        total = self.video_data.frame_count

        while self.video.isOpened() and idx < total:
            ok, img = self.video.read()
            if not ok or img is None:
                break

            bboxes, kpss = self._call_detect(img)

            # Post-filter by threshold 
            if bboxes is not None and len(bboxes) > 0:
                bboxes = np.asarray(bboxes, dtype=np.float32)
                # bboxes columns: x1,y1,x2,y2,score
                if bboxes.shape[1] >= 5:
                    keep = bboxes[:, 4] >= self.det_thresh
                    bboxes = bboxes[keep]
                    if kpss is not None:
                        try:
                            kpss = np.asarray(kpss)[keep]
                        except Exception:
                            # if it's a list and lengths mismatch, leave as-is
                            pass

            fr = FrameData(idx)
            if bboxes is not None and len(bboxes) > 0:
                for j in range(bboxes.shape[0]):
                    x1, y1, x2, y2 = map(int, np.round(bboxes[j, 0:4]))
                    w, h = x2 - x1, y2 - y1
                    keypoints = None
                    if kpss is not None and len(kpss) > j:
                        kp = np.asarray(kpss[j], dtype=np.float32)
                        if kp.shape == (5, 2):
                            keypoints = {
                                "left_eye":   (float(kp[0, 0]), float(kp[0, 1])),
                                "right_eye":  (float(kp[1, 0]), float(kp[1, 1])),
                                "nose":       (float(kp[2, 0]), float(kp[2, 1])),
                                "mouth_left": (float(kp[3, 0]), float(kp[3, 1])),
                                "mouth_right":(float(kp[4, 0]), float(kp[4, 1])),
                            }
                    face = Face(x1, y1, w, h, keypoints)
                    face.bind_to_frame(self.video_data.width, self.video_data.height)
                    fr.add_face(face)

            self.frames.append(fr)

            prog = float(idx) / max(1, total)
            if prog > self._next_tick:
                self._next_tick += self._tick
                print(f"{prog*100:2.0f}% ---- frame {idx:5d} ---- time {time.time()-self.start_time:10.4f} s")

            idx += 1

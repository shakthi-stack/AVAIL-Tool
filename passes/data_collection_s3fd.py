# code is based off of implementation in
# https://github.com/clcarwin/SFD_pytorch/blob/master/wider_eval_pytorch.py

import os, time, cv2, torch, numpy as np
from util.objects import *
from passes.data_collection_pass import DataCollectionPass
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast
from s3fd.net_s3fd import s3fd
from s3fd.bbox import *


class DataCollectionS3fd(DataCollectionPass):
    def __init__(self, video_data, frames, score_thr: float = 0.25, post_nms_score_thr: float = 0.5, iou_thr: float = 0.35, max_side = None,device='cuda', detect_large=True):
        super().__init__(video_data, frames)
        self.score_thr = score_thr
        self.post_nms_score_thr = post_nms_score_thr
        self.iou_thr = iou_thr
        self.max_side = max_side  # set None to disable any global downscale

        self.device = device
        self.net = s3fd()

        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.net.load_state_dict(torch.load(f'{current_dir}/../s3fd/s3fd_convert.pth'))
        try:
            self.net.to(self.device).float()
        except AssertionError as ae:
            print(ae)
            print(f'Casting detector to {self.device} failed.  Will fallback to cpu')
            self.device = 'cpu'
            self.net.to(self.device).float()

        self.net.eval()

        self.detect_large = detect_large
        torch.backends.cudnn.benchmark = True

    def execute(self):
        super().execute()

        index = 0
        next_tick = 0
        tick_interval = 0.05
        while self.video.isOpened() and index < self.video_data.frame_count:
            self.iterative_step(self.video, index)
            # quick command line progress bar
            if float(index) / self.video_data.frame_count > next_tick:
                next_tick += tick_interval
                print("{0:2.0f}% ---- frame {1:5.0f} ---- time {2:10.4f} s".format(
                    float(index) / self.video_data.frame_count * 100,
                    index,
                    time.time() - self.start_time))

            index += 1

    def iterative_step(self, video, index):
        # setup the frame object and read the image
        _, img = video.read()
        frame = FrameData(index)

        b1 = self.detect(self.net, img)
        b2 = self.flip_detect(self.net, img)
        if img.shape[0] * img.shape[1] * 4 > 3000 * 3000:
            b3 = np.zeros((1, 5))
        elif self.detect_large:
            try:
                b3 = self.scale_detect(self.net, img, scale=2, facesize=60)
            except:
                b3 = np.zeros((1, 5))
                self.detect_large = False
                print("Not enough memory to do double scale detection")
        else:
            b3 = np.zeros((1, 5))
        b4 = self.scale_detect(self.net, img, scale=0.5, facesize=100)
        bboxlist = np.concatenate((b1, b2, b3, b4))

        keep = nms(bboxlist, self.iou_thr)
        keep = keep[0:750]  # keep only max 750 boxes
        bboxlist = bboxlist[keep]
        # post‑NMS confidence filter
        bboxlist = bboxlist[bboxlist[:, 4] >= self.post_nms_score_thr]
        if bboxlist.size == 0:
            self.frames.append(frame)
            return

        for face in bboxlist:
            x = int(face[0])
            y = int(face[1])
            w = int(face[2] - x)
            h = int(face[3] - y)
            face_obj = Face(x, y, w, h)
            face_obj.bind_to_frame(self.video_data.width, self.video_data.height)
            frame.add_face(face_obj)
        # store this frame
        self.frames.append(frame)

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        if self.max_side and max(img.shape[:2]) > self.max_side:
            scale = self.max_side / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        img = (img - np.array([104, 117, 123], dtype=np.float32)).transpose(2, 0, 1)[None]
        return torch.from_numpy(img).float().to(self.device)
    
    def raw_s3fd(self, img_t: torch.Tensor):
        with torch.no_grad(), autocast():
            return self.net(img_t)
        
    def detect(self, net, img: np.ndarray) -> np.ndarray:
        img_t = self.preprocess(img)
        olist = self.raw_s3fd(img_t)

        # softmax class scores inplace on even tensors
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        bboxlist = []
        BB, CC, HH, WW = img_t.size()
        for i in range(int(len(olist) // 2)):
            ocls, oreg = olist[i * 2].cpu(), olist[i * 2 + 1].cpu()
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            anchor = stride * 4
            for Findex in range(FH * FW):
                windex, hindex = Findex % FW, Findex // FW
                # axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = float(ocls[0, 1, hindex, windex])
                if score < self.score_thr: continue
                loc = oreg[0, :, hindex, windex].view(1, 4).float()
                priors = torch.tensor([[stride / 2 + windex * stride,
                                         stride / 2 + hindex * stride,
                                         stride * 4,
                                         stride * 4]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append([x1, y1, x2, y2, score])

        if not bboxlist:
            return np.zeros((1, 5))
        
        bboxlist = np.array(bboxlist)
        # if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))
        return bboxlist

    def flip_detect(self, net, img):
        flipped = cv2.flip(img, 1)
        b = self.detect(net, flipped)
        if b.shape[0] == 1 and not b[0, 4]:  # no boxes
            return b
        bboxlist = b.copy()
        bboxlist[:, 0] = img.shape[1] - b[:, 2]
        bboxlist[:, 1] = b[:, 1]
        bboxlist[:, 2] = img.shape[1] - b[:, 0]
        bboxlist[:, 3] = b[:, 3]
        bboxlist[:, 4] = b[:, 4]
        return bboxlist

    def scale_detect(self, net, img, scale=2.0, facesize=None):
        scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        b = self.detect(net, scaled)
        b[:, :4] /= scale
        
        if scale > 1:
            mask = np.minimum(b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1) < facesize
        else:
            mask = np.maximum(b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1) > facesize
        b = b[mask]
        if b.size == 0:
            return np.zeros((1, 5))
        return b

# code is based off of implementation in
# https://github.com/clcarwin/SFD_pytorch/blob/master/wider_eval_pytorch.py

from util.objects import *
from passes.data_collection_pass import DataCollectionPass
import torch.nn.functional as F
from torch.autograd import Variable
from s3fd.net_s3fd import s3fd
from s3fd.bbox import *


class DataCollectionS3fd(DataCollectionPass):
    def __init__(self, video_data, frames, confidence=0.8, device='cuda', detect_large=True):
        super().__init__(video_data, frames)
        self.confidence = confidence
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
                b3 = self.scale_detect(self.net, img, scale=2, facesize=100)
            except:
                b3 = np.zeros((1, 5))
                self.detect_large = False
                print("Not enough memory to do double scale detection")
        else:
            b3 = np.zeros((1, 5))
        b4 = self.scale_detect(self.net, img, scale=0.5, facesize=100)
        bboxlist = np.concatenate((b1, b2, b3, b4))

        keep = nms(bboxlist, self.confidence)
        keep = keep[0:750]  # keep only max 750 boxes
        bboxlist = bboxlist[keep, :]

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

    def detect(self, net, img):
        img = img - np.array([104, 117, 123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,) + img.shape)

        with torch.no_grad():
            img = Variable(torch.from_numpy(img).float()).to(self.device)
        BB, CC, HH, WW = img.size()
        olist = net(img)

        bboxlist = []
        for i in range(int(len(olist) / 2)): olist[i * 2] = F.softmax(olist[i * 2])
        for i in range(int(len(olist) / 2)):
            ocls, oreg = olist[i * 2].data.cpu(), olist[i * 2 + 1].data.cpu()
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            anchor = stride * 4
            for Findex in range(FH * FW):
                windex, hindex = Findex % FW, Findex // FW
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
                if score < 0.05: continue
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))
        return bboxlist

    def flip_detect(self, net, img):
        img = cv2.flip(img, 1)
        b = self.detect(net, img)

        bboxlist = np.zeros(b.shape)
        bboxlist[:, 0] = img.shape[1] - b[:, 2]
        bboxlist[:, 1] = b[:, 1]
        bboxlist[:, 2] = img.shape[1] - b[:, 0]
        bboxlist[:, 3] = b[:, 3]
        bboxlist[:, 4] = b[:, 4]
        return bboxlist

    def scale_detect(self, net, img, scale=2.0, facesize=None):
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        b = self.detect(net, img)

        bboxlist = np.zeros(b.shape)
        bboxlist[:, 0] = b[:, 0] / scale
        bboxlist[:, 1] = b[:, 1] / scale
        bboxlist[:, 2] = b[:, 2] / scale
        bboxlist[:, 3] = b[:, 3] / scale
        bboxlist[:, 4] = b[:, 4]
        b = bboxlist
        if scale > 1:
            index = np.where(np.minimum(b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1) < facesize)[
                0]  # only detect small face
        else:
            index = np.where(np.maximum(b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1) > facesize)[
                0]  # only detect large face
        bboxlist = b[index, :]
        if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))
        return bboxlist

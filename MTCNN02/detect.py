from MTCNN02.net import *
from MTCNN02.data import tf
from PIL import Image, ImageDraw
import torch
from MTCNN02.utils import *

class Detector:
    def __init__(self):
        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load('pnet.pt'))  # 加载权重
        self.pnet.eval()

        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load('pnet.pt'))
        self.pnet.eval()

        self.pnet = PNet()
        self.pnet.load_state_dict(torch.load('pnet.pt'))
        self.pnet.eval()

    def __call__(self, img):
        boxes = self.detPnet(img)  # 放入图片，返回一组框
        if boxes is None: return []  # 如果不存在人脸，返回一个框
        # return boxes


        boxes = self.detRnet(img, boxes)
        if boxes is None: return []

        boxes = self.detOnet(img, boxes)
        if boxes is None: return []

        return boxes

    def detPnet(self, img):
        w, h = img.size  # 取出导入的要检测的图片的宽和高
        scale = 1  # 缩放比例，循环之后缩小
        img_scale = img  # 存储缩放图，临时变量
        min_side = min(w, h)  # 如果是矩形，取出最短的边


        _boxes = []
        while min_side > 12:
            _img_scale = tf(img_scale)
            y = self.pnet(_img_scale[None, ...])  # 由于导入的是一张图片，所以要增加一个批次的维度
            y = y.cpu().detach()

            torch.sigmoid_(y[:, 0, ...])
            c = y[0, 0]  # 置信度去掉前面通道
            c_mask = c > 0.6  # 得到置信度大于该值的框
            idxs = c_mask.nonzero()  # 人脸索引
            _x1, _y1 = idxs[:, 1] * 2, idxs[:, 0] * 2  # 2为组合卷积的步长，求出坐标
            _x2, _y2 = _x1 + 12, _y1 + 12


            # 反算过程，应该与前面gen_data相对应
            p = y[0, 1:, c_mask]
            # print(p[0, :], 333)
            x1 = (_x1 - p[0, :] * 12) / scale
            y1 = (_y1 - p[1, :] * 12) / scale
            x2 = (_x2 - p[2, :] * 12) / scale
            y2 = (_y2 - p[3, :] * 12) / scale

            cc = y[0, 0, c_mask]

            _boxes.append(torch.stack([x1, y1, x2, y2, cc], dim=1))  # stack增加维度，把置信度放在最后一个维度可以避免再次修改utils代码

            # 下采样过程
            scale *= 0.702
            _w, _h = int(w * scale), int(h * scale)
            img_scale = img_scale.resize((_w, _h))
            min_side = min(_w, _h)

        boxes = torch.cat(_boxes, dim=0)  # 把框合并

        return nms(boxes.cpu().detach().numpy(), 0.8)  # 值越小保留的框越少

    def detRnet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 24)
        return nms(_boxes, 0.9)

    def detOnet(self, img, boxes):
        _boxes = self._rnet_onet(img, boxes, 48)
        _boxes = nms(_boxes, 0.9)
        _boxes = nms(_boxes, 0.8, is_min=True)  # 去除重叠的框
        return _boxes

    def _rnet_onet(self, img, boxes, s):
        imgs = []
        for box in boxes:
            _img = img.crop(box[0:4])
            _img = _img.resize((s, s))
            imgs.append(tf(_img))
        _imgs = torch.stack(imgs, dim=0)
        if s == 24:
            y = self.rnet(_imgs)
        else:
            y = self.onet(_imgs)

        y = y.cpu().detach()
        torch.sigmoid_(y[:, 0])
        y = y.numpy()

        c_mask = y[:, 0] > 0.8
        _boxes = boxes[c_mask]
        _y = y[c_mask]
        # print(_y)

        _w, _h = _boxes[:, 2] - _boxes[:, 0], _boxes[:, 3] - _boxes[:, 1]
        x1 = _boxes[:, 0] + _y[:, 1] * _w
        y1 = _boxes[:, 1] + _y[:, 2] * _h
        x2 = _boxes[:, 2] + _y[:, 3] * _w
        y2 = _boxes[:, 3] + _y[:, 4] * _h
        cc = _y[:, 0]

        _boxes = np.stack([x1, y1, x2, y2, cc], axis=1)

        return _boxes
        # print(_boxes)

if __name__ == '__main__':
    img = Image.open('12.jpg')
    detector = Detector()
    boxes = detector(img)
    drawing = ImageDraw.Draw(img)
    for i, box in enumerate(boxes):
        drawing.rectangle((box[0], box[1], box[2], box[3]), None, 'red')
    img.show()
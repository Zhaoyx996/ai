# 用数据集生成样本，3中尺寸3种标签
import os
from PIL import Image
import numpy as np
import MTCNN02.utils_test


# 创建生成数据的类
class GenData:
    def __init__(self, root, img_size):
        self.img_size = img_size  # 图片尺寸
        # 样本路径，标签路径
        self.positive_image_dir = f"{root}/{img_size}/positive"
        self.negative_image_dir = f"{root}/{img_size}/negative"
        self.part_image_dir = f"{root}/{img_size}/part"

        self.positive_label = f"{root}/{img_size}/positive.txt"
        self.negative_label = f"{root}/{img_size}/negative.txt"
        self.part_label = f"{root}/{img_size}/part.txt"
        # 判断文件存在与否，否创建文件
        if not os.path.exists(self.positive_image_dir):
            os.makedirs(self.positive_image_dir)

        if not os.path.exists(self.negative_image_dir):
            os.makedirs(self.negative_image_dir)

        if not os.path.exists(self.part_image_dir):
            os.makedirs(self.part_image_dir)
        # 参数路径，标签
        self.anno_box_path = r"D:/CelebA_data/CelebA/Anno/list_bbox_celeba.txt"
        self.anno_landmark_path = r"D:/CelebA_data/CelebA/Anno/list_landmarks_celeba.txt"
        self.img_path = r"D:/CelebA_data/CelebA/Img/img_celeba.7z/img_celeba"

    # 循环生成数据12
    def run(self, epoch):
        positive_label_txt = open(self.positive_label, "w")  # w写文件，后面要关闭
        negative_label_txt = open(self.negative_label, "w")
        part_label_txt = open(self.part_label, "w")
        # 计数器重置
        positive_count = 0
        negative_count = 0
        part_count = 0
        # 开启循环
        for _ in range(epoch):
            for i, line in enumerate(open(self.anno_box_path, "r")):  # r代表读取
                if i < 2: continue  # 小于2不处理
                print(line)
                strs = line.split()  # 字符串分解
                img = Image.open(f"{self.img_path}/{strs[0]}")  # 打开图片
                x, y, w, h = int(strs[1]), int(strs[2]), int(strs[3]), int(strs[4])  # 得到实际框坐标
                # 五官
                # px1, py1 = int(strs[5]), int(strs[6])
                # px2, py2 = int(strs[7]), int(strs[8])
                # px3, py3 = int(strs[9]), int(strs[10])
                # px4, py4 = int(strs[11]), int(strs[12])
                # px5, py5 = int(strs[13]), int(strs[14])

                # 矫正样本
                x1, y1, x2, y2 = int(x + w * 0.12), int(y + h * 0.1), int(x + w * 0.9), int(y + h * 0.85)  #
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # 筛选并防止数据尺寸过小
                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                cx, cy = int(x + w / 2), int(y + h / 2)  # 人脸中心
                # 浮动框，浮动随机数randint，中心点左右偏移，
                _cx, _cy = cx + np.random.randint(-w * 0.2, w * 0.2), cy + np.random.randint(-h * 0.2, h * 0.2)
                _w, _h = w + np.random.randint(-w * 0.2, w * 0.2), h + np.random.randint(-h * 0.2, h * 0.2)
                _x1, _y1, _x2, _y2 = int(_cx - _w / 2), int(_cy - _h / 2), int(_cx + _w / 2), int(_cy + _h / 2)
                # 筛选，防止后面被除数为0
                if _w == 0 or _h == 0:
                    continue
                # print(x, y, w, h)
                # 截图查看
                clip_img = img.crop([_x1, _y1, _x2, _y2])  # 通过框的坐标，截出图片
                clip_img = clip_img.resize((self.img_size, self.img_size))  # 设置截图大小，12， 24， 48
                # clip_img.show()
                # iou比较
                iou = MTCNN02.utils_test.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))

                if iou > 0.6:  # 原定0.65
                    clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")
                    _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                _y2 - y2) / _h
                    # offset_px1, offset_py1 = (px1 - x1) / _w, (py1 - y1) / _h
                    # offset_px2, offset_py2 = (px2 - x1) / _w, (py2 - y1) / _h
                    # offset_px3, offset_py3 = (px3 - x1) / _w, (py3 - y1) / _h
                    # offset_px4, offset_py4 = (px4 - x1) / _w, (py4 - y1) / _h
                    # offset_px5, offset_py5 = (px5 - x1) / _w, (py5 - y1) / _h

                    positive_label_txt.write(f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} \n")
                    # {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                    positive_label_txt.flush()
                    positive_count += 1
                elif iou > 0.4:
                    clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                    _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                _y2 - y2) / _h
                    # offset_px1, offset_py1 = (px1 - x1) / _w, (py1 - y1) / _h
                    # offset_px2, offset_py2 = (px2 - x1) / _w, (py2 - y1) / _h
                    # offset_px3, offset_py3 = (px3 - x1) / _w, (py3 - y1) / _h
                    # offset_px4, offset_py4 = (px4 - x1) / _w, (py4 - y1) / _h
                    # offset_px5, offset_py5 = (px5 - x1) / _w, (py5 - y1) / _h
                    part_label_txt.write(f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off} \n")
                    # {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                    part_label_txt.flush()
                    part_count += 1
                elif iou < 0.3:
                    clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")

                    negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n")
                    negative_label_txt.flush()
                    negative_count += 1

                # 生成负样本
                w, h = img.size  # 取出尺寸
                _x1, _y1 = np.random.randint(0, w), np.random.randint(0, h)  # 随机起始点
                _w, _h = np.random.randint(0, w - _x1), np.random.randint(0, h - _y1)

                if _w == 0 or _h == 0:
                    continue
                _x2, _y2 = _x1 + w, _y1 + _h

                clip_img = img.crop([_x1, _y1, _x2, _y2])
                clip_img = clip_img.resize((self.img_size, self.img_size))  # 更改图片尺寸

                iou = MTCNN02.utils_test.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))
                if iou > 0.65:
                    clip_img.save(f"{self.positive_image_dir}/{positive_count}.jpg")  # 将截图保存
                    _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                _y2 - y2) / _h
                    # offset_px1, offset_py1 = (px1 - x1) / _w, (py1 - y1) / _h
                    # offset_px2, offset_py2 = (px2 - x1) / _w, (py2 - y1) / _h
                    # offset_px3, offset_py3 = (px3 - x1) / _w, (py3 - y1) / _h
                    # offset_px4, offset_py4 = (px4 - x1) / _w, (py4 - y1) / _h
                    # offset_px5, offset_py5 = (px5 - x1) / _w, (py5 - y1) / _h
                    # 存文件名字，标签1，坐标
                    positive_label_txt.write(f"{positive_count}.jpg 1 {_x1_off} {_y1_off} {_x2_off} {_y2_off} \n")
                    # {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                    positive_label_txt.flush()
                    positive_count += 1
                elif iou > 0.4:
                    clip_img.save(f"{self.part_image_dir}/{part_count}.jpg")
                    _x1_off, _y1_off, _x2_off, _y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (
                                _y2 - y2) / _h
                    # offset_px1, offset_py1 = (px1 - x1) / _w, (py1 - y1) / _h
                    # offset_px2, offset_py2 = (px2 - x1) / _w, (py2 - y1) / _h
                    # offset_px3, offset_py3 = (px3 - x1) / _w, (py3 - y1) / _h
                    # offset_px4, offset_py4 = (px4 - x1) / _w, (py4 - y1) / _h
                    # offset_px5, offset_py5 = (px5 - x1) / _w, (py5 - y1) / _h
                    part_label_txt.write(f"{part_count}.jpg 2 {_x1_off} {_y1_off} {_x2_off} {_y2_off} \n")
                    # {offset_px1} {offset_py1} {offset_px2} {offset_py2} {offset_px3} {offset_py3} {offset_px4} {offset_py4} {offset_px5} {offset_py5}\n")
                    part_label_txt.flush()
                    part_count += 1
                elif iou < 0.3:
                    clip_img.save(f"{self.negative_image_dir}/{negative_count}.jpg")

                    negative_label_txt.write(f"{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \n")
                    negative_label_txt.flush()
                    negative_count += 1

        # 执行完后关闭文件资源
        positive_label_txt.close()
        negative_label_txt.close()
        part_label_txt.close()


if __name__ == '__main__':
    genData = GenData(r"D:\mtcnn_data", 48)
    genData.run(10)
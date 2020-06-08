# 导入模块
import os
import numpy as np
from PIL import Image
import MTCNN02.utils


# 创建gen_data类，创建三种大小三种标签的数据
class Gen_data:
    # 传入根目录和图片大小两个参数
    def __init__(self, root, img_size):
        # 传入图片大小
        self.img_size = img_size

        # 传入三种样本图片数据的文件夹地址
        self.positive_img = f'{root}/{img_size}/positive_img'
        self.negative_img = f'{root}/{img_size}/negative_img'
        self.part_img = f'{root}/{img_size}/part_img'
        # 传入三种图片样本所对应的标签的txt文件地址，为后面的写入标签训练做准备
        self.positive_label = f'{root}/{img_size}/positive_label.txt'
        self.negative_label = f'{root}/{img_size}/negative_label.txt'
        self.part_label = f'{root}/{img_size}/part_label.txt'

        # 如果图片数据所在的文件夹不存在，就创建三个
        if not os.path.exists(self.positive_img):
            os.makedirs(self.positive_img)

        if not os.path.exists(self.negative_img):
            os.makedirs(self.negative_img)

        if not os.path.exists(self.part_img):
            os.makedirs(self.part_img)

        # 传入目标图片，目标框，关键点所在的文件夹地址
        self.img_dir = 'D:/CelebA_data/CelebA/Img/img_celeba.7z/img_celeba'
        self.box_dir = 'D:/CelebA_data/CelebA/Anno/list_bbox_celeba.txt'
        self.landmark_dir = 'D:/CelebA_data/CelebA/Anno/list_landmarks_celeba.txt'

    # 创建样本集
    def run(self, epoch):
        # 打开三种标签的txt文件
        positive_label_txt = open(self.positive_label, 'w')  # w是写入的意思，后面会关闭
        part_label_txt = open(self.part_label, 'w')
        negative_label_txt = open(self.negative_label, 'w')

        # 计数器重置
        positive_count = 0
        negative_count = 0
        part_count = 0

        # 开始循环
        for _ in range(epoch):
            for i, line_box in enumerate(open(self.box_dir, 'r')):   # r代表只读

                if i < 2:
                    continue  # 第一第二行不操作

                strs = line_box.split()   # 字符串分解
                img = Image.open(f'{self.img_dir}/{strs[0]}')
                x, y, w, h = int(strs[1]), int(strs[2]), int(strs[3]), int(strs[4])   # 提取出坐标和长宽

                # 目标框的坐标不够理想，调整一下位置
                x1, y1, x2, y2 = int(x + 0.12*w), int(y + 0.1*h), int(x + 0.9*w), int(y + 0.85*h)

                # 调整后的x, y, w, h
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                # 防止生成不符合条件的数据，使生成挂掉
                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                # 找出中心点的坐标
                cx, cy = int(x + w/2), int(y + h/2)

                # 用randint函数浮动一下中心点的坐标和宽度高度
                _cx, _cy = cx + np.random.randint(-0.2*w, 0.2*w), cy + np.random.randint(-0.2*h, 0.2*h)
                _w, _h = w + np.random.randint(-0.2*w, 0.2*w), h + np.random.randint(-0.2*h, 0.2*h)

                # 浮动后新框的坐标
                _x1, _y1, _x2, _y2 = int(_cx - _w/2), int(_cy - _h/2), int(_cx + _w/2), int(_cy + _h/2)

                # 筛选，防止后面被除数为0
                if _w == 0 or _h == 0:
                    continue

                # 将浮动后图像切割出来
                img_clip = img.crop([_x1, _y1, _x2, _y2])

                # 变换图片大小为12，24，48
                img_clip = img_clip.resize((self.img_size, self.img_size))

                iou = MTCNN02.utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))  # 切割后的样本图片是多张，所以需要多加一个维度

                # 根据iou分类保存图片
                if iou > 0.6:
                    img_clip.save(f'{self.positive_img}/{positive_count}.jpg')

                    # 计算出偏移量，这里怎么计算的，侦测时就怎么反算回去;应该除以建议框的高和宽
                    x1_off, y1_off, x2_off, y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (_y2 - y2) / _h

                    # 把偏移量的值写入txt文件
                    positive_label_txt.write(f'{positive_count}.jpg 1 {x1_off} {x2_off} {y1_off} {y2_off}\n')
                    positive_label_txt.flush()
                    positive_count += 1


                elif iou > 0.4:
                    img_clip.save(f'{self.part_img}/{part_count}.jpg')

                    # 计算出偏移量，这里怎么计算的，侦测时就怎么反算回去;应该除以建议框的高和宽
                    x1_off, y1_off, x2_off, y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (_y2 - y2) / _h

                    # 把偏移量的值写入txt文件
                    part_label_txt.write(f'{part_count}.jpg 2 {x1_off} {x2_off} {y1_off} {y2_off}\n')
                    part_label_txt.flush()
                    part_count += 1

                elif iou < 0.3:
                    img_clip.save(f'{self.negative_img}/{negative_count}.jpg')

                    # 把偏移量的值写入txt文件
                    negative_label_txt.write(f'{negative_count}.jpg 0 0 0 0 0\n')
                    negative_label_txt.flush()
                    negative_count += 1

                # 生成负样本集
                w, h = img.size

                # 随机生成0~w,0~h的数来作为新框左上角的坐标
                _x, _y = np.random.randint(0, w), np.random.randint(0, h)

                # 由于随机生成的框的大小不应该超过原图像的范围，所以随机生成的框的宽和高应该是0到剩下的距离
                _w, _h = np.random.randint(0, w - _x), np.random.randint(0, h - _y)
                _x1, _y1, _x2, _y2 = _x, _y, _x + _w, _y + _h

                if _w == 0 or _h == 0:
                    continue

                img_clip = img.crop([_x1, _y1, _x2, _y2])
                img_clip = img_clip.resize((self.img_size, self.img_size))

                iou = MTCNN02.utils.iou(np.array([x1, y1, x2, y2]), np.array([[_x1, _y1, _x2, _y2]]))  # 也要加上iou

                # 根据iou分类保存图片
                if iou > 0.6:
                    img_clip.save(f'{self.positive_img}/{positive_count}.jpg')

                    # 计算出偏移量，这里怎么计算的，侦测时就怎么反算回去;应该除以建议框的高和宽
                    x1_off, y1_off, x2_off, y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (_y2 - y2) / _h

                    # 把偏移量的值写入txt文件
                    positive_label_txt.write(f'{positive_count}.jpg 1 {x1_off} {x2_off} {y1_off} {y2_off}\n')
                    positive_label_txt.flush()
                    positive_count += 1


                elif iou > 0.4:
                    img_clip.save(f'{self.part_img}/{part_count}.jpg')

                    # 计算出偏移量，这里怎么计算的，侦测时就怎么反算回去;应该除以建议框的高和宽
                    x1_off, y1_off, x2_off, y2_off = (_x1 - x1) / _w, (_y1 - y1) / _h, (_x2 - x2) / _w, (_y2 - y2) / _h

                    # 把偏移量的值写入txt文件
                    part_label_txt.write(f'{part_count}.jpg 2 {x1_off} {x2_off} {y1_off} {y2_off}\n')
                    part_label_txt.flush()
                    part_count += 1

                elif iou < 0.3:
                    img_clip.save(f'{self.negative_img}/{negative_count}.jpg')

                    # 把偏移量的值写入txt文件
                    negative_label_txt.write(f'{negative_count}.jpg 0 0 0 0 0\n')
                    negative_label_txt.flush()
                    negative_count += 1

        positive_label_txt.close()
        part_label_txt.close()
        negative_label_txt.close()


if __name__ == '__main__':
    GenData = Gen_data('D:/mtcnn_data', 48)
    GenData.run(10)

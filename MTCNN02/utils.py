import numpy as np


# [x1, y1, x2, y2] 框的坐标格式
def iou(box, boxes, is_min=False):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 截取出来的框是多个，所以在前面多一个维度

    x1 = np.maximum(box[0], boxes[:, 0])  # 取目标框和建议框的最大值，取得交集的坐标
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # 取出交集的长和宽，计算面积，由于存在负数的情况，所以应该加上0比大小
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    inter = w*h

    if is_min:
        return inter/np.minimum(box_area, boxes_area)
    else:
        return inter/(boxes_area + box_area - inter)


def nms(boxes, threshold, is_min=False):  # 传入框，阈值
    if boxes.shape[0] == 0:return np.array([])  # 如果不存在框，返回框
    _boxes = boxes[(-boxes[:, 4]).argsort()]  # 把框之间的iou从大到小排序

    r_boxes = []

    while _boxes.shape[0] > 1:  # 当所有的框多于一个时
        a_box = _boxes[0]  # 取出第一个框
        b_boxes = _boxes[1:]  # 取出剩余的框
        r_boxes.append(a_box)  # 把第一个框加进去
        _boxes = b_boxes[iou(a_box, b_boxes, is_min) < threshold]

    if _boxes.shape[0] > 0:  # 如果剩下一个，保留
        r_boxes.append(_boxes[0])  # 保留剩余的框

    return np.array(r_boxes)  # 返回留下来的框

if __name__ == '__main__':
    box = np.array([1, 1, 3, 3])
    boxes = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [2, 2, 4, 4]])
    y = iou(box, boxes, False)
    print(y)

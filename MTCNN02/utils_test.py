import numpy as np

#[x1,y1,x2,y2]，，对应放的方式
def iou(box, boxes, is_min=False): #最小值的IOU
    #框面积计算
    box_area = (box[2]-box[0])*(box[3]-box[1])
    boxes_area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) #两个维度，一维是批次，二位参数

    x1 = np.maximum(box[0], boxes[:,0]) # 左上角最大值
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])

    w = np.maximum(0, x2-x1)
    h = np.maximum(0, y2-y1)
    inter = w*h # 交集

    if is_min:
        return inter/np.minimum(box_area, boxes_area)
    else:
        return inter/(box_area+boxes_area-inter)


#nms非极大值抑制，传一组框，
def nms(boxes, threshold, is_min=False): #传框,阈值
    if boxes.shape[0] == 0: return np.array([])
    _boxes = boxes[(-boxes[:, 4]).argsort()] #置信度进行排序，负号从大到小，argsort得到索引

    r_boxes = []
    while _boxes.shape[0]>1: #框有多个时比较
        a_box = _boxes[0] #第一框保留
        b_boxes = _boxes[1:] #其他的所有框

        r_boxes.append(a_box) #每循环一次，添加一个框，第一框加进去
        _boxes = b_boxes[iou(a_box, b_boxes, is_min) < threshold] #小于阈值

    if _boxes.shape[0] > 0: #如果剩下一个，保留
        r_boxes.append(_boxes[0])

    return np.array(r_boxes) #返回留下来的框

if __name__ == '__main__':
    box = np.array([1, 1, 3, 3])
    boxes = np.array([[1, 1, 3, 3], [3, 3, 5, 5], [2, 2, 4, 4]])
    y = iou(box, boxes, True)
    print(y)
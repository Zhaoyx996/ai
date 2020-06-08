import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

tf = transforms.Compose([transforms.ToTensor()])  # 将数据转化为标准正态分布

class Dataset(Dataset):
    def __init__(self, root, img_size):
        self.dataset = []
        self.root_dir = f'{root}/{img_size}'  # 申请资源

        with open(f'{self.root_dir}/positive_label.txt', 'r') as f:
            self.dataset.extend(f.readlines())

        with open(f'{self.root_dir}/negative_label.txt', 'r') as f:
            self.dataset.extend((f.readlines()))

        with open(f'{self.root_dir}/part_label.txt', 'r') as f:
            self.dataset.extend((f.readlines()))

    def __len__(self):
        # 取长度
        return len(self.dataset)

    def __getitem__(self, index):
        # 取索引
        data = self.dataset[index]
        # 字符串分解
        strs = data.split()

        if strs[1] == '1':
            img_path = f'{self.root_dir}/positive_img/{strs[0]}'

        if strs[1] == '0':
            img_path = f'{self.root_dir}/negative_img/{strs[0]}'

        if strs[1] == '2':
            img_path = f'{self.root_dir}/part_img/{strs[0]}'

        img_data = tf(Image.open(img_path))

        c, x1, y1, x2, y2 = np.float(strs[1]), np.float(strs[2]), np.float(strs[3]), np.float(strs[4]), np.float(strs[5])

        return img_data, np.array([c, x1, y1, x2, y2], dtype=np.float32)


if __name__ == '__main__':
    dataset = Dataset('D:/mtcnn_data', 12)
    print(dataset[0])
from MTCNN02.data import *
from MTCNN02.net import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim

DEVICE = 'cuda:0'

class Train:
    def __init__(self, root, img_size):

        self.train_data = Dataset(root, img_size)
        self.train_dataLoader = DataLoader(self.train_data, batch_size=2000, shuffle=True)

        self.img_size = img_size

        if img_size == 12:
            self.net = PNet()

        if img_size == 24:
            self.net = RNet()

        if img_size == 48:
            self.net = ONet()

        self.net.to(DEVICE)

        self.opt = optim.Adam(self.net.parameters())

    def __call__(self, epochs):

        for epoch in range(epochs):

            self.net.train()
            for i, (imgs, tags) in enumerate(self.train_dataLoader):
                imgs, tags = imgs.to(DEVICE), tags.to(DEVICE)

                self.net.train()
                self.net.eval()

                predict = self.net(imgs)

                if self.img_size == 12:
                    predict = predict.reshape(-1, 5)  # p网络是全卷积结构，所以要变化形状

                torch.sigmoid_(predict[:, 0])

                # 用正样本和负样本训练置信度
                c_mask = tags[:, 0] < 2
                c_pre = predict[c_mask]  # 取出正样本和负样本的预测值
                c_tag = tags[c_mask]  # 取出正样本和负样本的置信度标签值
                loss_c = torch.mean((c_tag[:, 0] - c_pre[:, 0]) ** 2)  # 置信度在第一个维度，做mse

                # 用部分样本和负样本训练坐标
                off_mask = tags[:, 0] > 0
                off_pre = predict[off_mask]
                off_tag = tags[off_mask]
                loss_off = torch.mean((off_tag[:, 1:] - off_pre[:, 1:]) ** 2)

                loss = loss_c + loss_off

                # 反向传播算法，套路
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                print(loss)
                print(epoch)

                # 保存权重
                if self.img_size == 12:
                    torch.save(self.net.state_dict(), 'pnet.pt')

                elif self.img_size == 24:
                    torch.save(self.net.state_dict(), 'rnet.pt')

                else:
                    torch.save(self.net.state_dict(), 'onet.pt')

if __name__ == '__main__':
    train = Train("D:/mtcnn_data", 48)
    train(1000)  # 训练轮次





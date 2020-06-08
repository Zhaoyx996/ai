import torch
from torch import nn

class PNet(nn.Sequential):  # 这样继承可以省略前算过程,pnet是一个全卷积结构
    def __init__(self):
        super(PNet, self).__init__(
            nn.Conv2d(3, 10, 3, padding=1),
            nn.PReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(10, 16, 3),
            nn.PReLU(),

            nn.Conv2d(16, 32, 3),
            nn.PReLU(),

            nn.Conv2d(32, 5, 1)
        )


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 28, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(28, 48, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 64, 2)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(3*3*64, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h.reshape(-1, 3*3*64)  # 要变换维度才能继续计算
        h = self.output_layer(h)
        return h

class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 2)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(3*3*128, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = h.reshape(-1, 3*3*128)
        h = self.output_layer(h)
        return h


if __name__ == '__main__':
    # pNet = PNet()
    # pin = torch.randn(1, 3, 12, 12)
    # py = pNet(pin)
    # print(py.shape)

    rNet = RNet()
    rin = torch.randn(1, 3, 24, 24)
    ry = rNet(rin)
    print(ry.shape)

    # oNet = ONet()
    # oin = torch.randn(1, 3, 48, 48)
    # oy = oNet(oin)
    # print(oy.shape)
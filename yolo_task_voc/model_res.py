import torchvision.models as models
import torch
import torch.nn as nn


class YoloV1(nn.Module):
    def __init__(self, S, B, C):
        super(YoloV1, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)

        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 496)
        self.dropout = nn.Dropout(0.5)
        self.leakyrelu_fc = nn.LeakyReLU(0.1)
        self.linear = nn.Linear(496, S * S * (C + B * 5))

    def forward(self, x):

        out = self.resnet18(x)

        out = self.dropout(out)
        out = self.leakyrelu_fc(out)
        out = self.linear(out)

        return out


def main():

    x = torch.randn((13, 3, 448, 448))

    resnet18 = models.resnet18(pretrained=True)
    resnet18.eval()
    y_p = resnet18(x)
    print(y_p.shape)

    yolov1 = YoloV1(S=7, B=2, C=20)
    yolov1.eval()
    y_p = yolov1(x)
    print(y_p.shape)


if __name__ == '__main__':
    main()

import torchvision.models as models
import torch.nn as nn
import torch


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        _resnet = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*_resnet)  # Resnet without the fc layer

    def forward(self, x):
        return self.resnet(x)


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CnnBlock(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1),
            CnnBlock(2 * in_channels, (num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)  # [p, x, y, w, h]
        )

    def forward(self, x):
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        # N x 3 x 13 x 13 x 5+num_classes (3 is number of anchors)
        # N x 3 x 26 x 26 x 5+num_classes (3 is number of anchors)
        # N x 3 x 52 x 52 x 5+num_classes (3 is number of anchors)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [nn.Sequential(CnnBlock(channels, channels // 2, kernel_size=1, stride=1, padding=0),
                                          CnnBlock(channels // 2, channels, kernel_size=3, stride=1, padding=1))]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()


class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.feature_extractor = FeatureExtractor()
        # (512, 1, 1)
        self.cnn_block_1 = CnnBlock(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        # (1024, 3, 1)
        self.cnn_block_2 = CnnBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        # S
        self.resudual_1 = ResidualBlock()
        self.scale_prediction_1 = ScalePrediction(in_channels=1024, num_classes=20)

        self.cnn_block_3 = CnnBlock(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cnn_block_1(x)
        x = self.cnn_block_2(x)
        out_1 = self.scale_prediction_1(x)
        x = self.cnn_block_3(x)
        return (out_1)

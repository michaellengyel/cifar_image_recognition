import torch
import torch.nn as nn
import torchvision.models as models


# Tuple: (out_channels, kernel_size, stride)
# List: [Block, num_of_repeats]
# "S: Scale prediction
# "U": Upsampling
config = [
    # Start of Darknet-53
    #(32, 3, 1),
    #(64, 3, 2),
    #["B", 1],
    #(128, 3, 2),
    #["B", 2],
    #(256, 3, 2),
    #["B", 8],
    #(512, 3, 2),
    #["B", 8],
    #(1024, 3, 2),
    #["B", 4],
    #End of Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S"
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm_and_activation=True, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not batchnorm_and_activation, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.use_batchnorm_and_activation = batchnorm_and_activation

    def forward(self, x):
        if self.use_batchnorm_and_activation:
            return self.leakyrelu(self.batchnorm(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [nn.Sequential(CNNBlock(channels, channels//2, kernel_size=1), CNNBlock(channels//2, channels, kernel_size=3, padding=1))]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3, batchnorm_and_activation=False, kernel_size=1)  # [p, x, y, w, h]
        )

        self.num_classes = num_classes

    def forward(self, x):
        return self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        # N x 3 x 13 x 13 x 5+num_classes (3 is number of anchors)
        # N x 3 x 26 x 26 x 5+num_classes (3 is number of anchors)
        # N x 3 x 52 x 52 x 5+num_classes (3 is number of anchors)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # resnet = models.resnet101(pretrained=True)
        # resnet = models.resnet152(pretrained=True)
        # resnet = models.mobilenet_v3_large(pretrained=True)
        resnet = models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        _resnet = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*_resnet)  # Resnet without the fc layer
        self.conv = CNNBlock(512, 2*512, kernel_size=3, padding=1)

    def forward(self, x):

        route_connections = []

        for layer in self.resnet.children():
            x = layer(x)
            if x.shape[-1] == 52:  # YOLO head scip connection
                route_connections.append(x)
            elif x.shape[-1] == 26:  # YOLO head scip connection
                route_connections.append(x)

        x = self.conv(x)
        return x, route_connections


class YoloV3(nn.Module):
    def __init__(self, in_channels=2*512, num_classes=90):
        super(YoloV3, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feature_extractor = FeatureExtractor()
        self.layers = self._create_conv_layers()

    def forward(self, x):

        x, route_connections = self.feature_extractor(x)

        outputs = []

        for index, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0))
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 2  # We want to concatenate just after the up-sample

        return layers


def main():

    num_classes = 90
    IMAGE_SIZE = 416
    model = YoloV3(num_classes=num_classes)
    x = torch.randn((16, 3, IMAGE_SIZE, IMAGE_SIZE))
    print(model.parameters())
    yp = model(x)
    print(len(yp))
    print(yp[0].shape)
    print(yp[1].shape)
    print(yp[2].shape)


if __name__ == '__main__':
    main()

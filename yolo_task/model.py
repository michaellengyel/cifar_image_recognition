import torch
import torch.nn as nn


class YoloV1(nn.Module):
    def __init__(self):
        super(YoloV1, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm_0 = nn.BatchNorm2d(64)
        self.leakyrelu_0 = nn.LeakyReLU(0.1)

        self.maxpool_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(192)
        self.leakyrelu_1 = nn.LeakyReLU(0.1)

        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_2 = nn.BatchNorm2d(128)
        self.leakyrelu_2 = nn.LeakyReLU(0.1)

        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_3 = nn.BatchNorm2d(256)
        self.leakyrelu_3 = nn.LeakyReLU(0.1)

        self.conv_4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_4 = nn.BatchNorm2d(128)
        self.leakyrelu_4 = nn.LeakyReLU(0.1)

        self.conv_5 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=0, bias=False)
        self.batchnorm_5 = nn.BatchNorm2d(512)
        self.leakyrelu_5 = nn.LeakyReLU(0.1)

        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_6 = nn.BatchNorm2d(256)
        self.leakyrelu_6 = nn.LeakyReLU(0.1)

        self.conv_7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_7 = nn.BatchNorm2d(512)
        self.leakyrelu_7 = nn.LeakyReLU(0.1)

        self.conv_8 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_8 = nn.BatchNorm2d(256)
        self.leakyrelu_8 = nn.LeakyReLU(0.1)

        self.conv_9 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_9 = nn.BatchNorm2d(512)
        self.leakyrelu_9 = nn.LeakyReLU(0.1)

        self.conv_10 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_10 = nn.BatchNorm2d(256)
        self.leakyrelu_10 = nn.LeakyReLU(0.1)

        self.conv_11 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_11 = nn.BatchNorm2d(512)
        self.leakyrelu_11 = nn.LeakyReLU(0.1)

        self.conv_12 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_12 = nn.BatchNorm2d(256)
        self.leakyrelu_12 = nn.LeakyReLU(0.1)

        self.conv_13 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_13 = nn.BatchNorm2d(512)
        self.leakyrelu_13 = nn.LeakyReLU(0.1)

        self.conv_14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_14 = nn.BatchNorm2d(512)
        self.leakyrelu_14 = nn.LeakyReLU(0.1)

        self.conv_15 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_15 = nn.BatchNorm2d(1024)
        self.leakyrelu_15 = nn.LeakyReLU(0.1)

        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_16 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_16 = nn.BatchNorm2d(512)
        self.leakyrelu_16 = nn.LeakyReLU(0.1)

        self.conv_17 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_17 = nn.BatchNorm2d(1024)
        self.leakyrelu_17 = nn.LeakyReLU(0.1)

        self.conv_18 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False)
        self.batchnorm_18 = nn.BatchNorm2d(512)
        self.leakyrelu_18 = nn.LeakyReLU(0.1)

        self.conv_19 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_19 = nn.BatchNorm2d(1024)
        self.leakyrelu_19 = nn.LeakyReLU(0.1)

        self.conv_20 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_20 = nn.BatchNorm2d(1024)
        self.leakyrelu_20 = nn.LeakyReLU(0.1)

        self.conv_21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.batchnorm_21 = nn.BatchNorm2d(1024)
        self.leakyrelu_21 = nn.LeakyReLU(0.1)

        self.conv_22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_22 = nn.BatchNorm2d(1024)
        self.leakyrelu_22 = nn.LeakyReLU(0.1)

        self.conv_23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm_23 = nn.BatchNorm2d(1024)
        self.leakyrelu_23 = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(1024 * 7 * 7, 496)
        self.dropout = nn.Dropout(0.5)
        self.leakyrelu_fc = nn.LeakyReLU(0.1)
        self.linear_1 = nn.Linear(496, 7 * 7 * (20 + 2 * 5))

    def forward(self, x):

        out = self.conv_0(x)
        out = self.batchnorm_0(out)
        out = self.leakyrelu_0(out)

        out = self.maxpool_0(out)

        out = self.conv_1(out)
        out = self.batchnorm_1(out)
        out = self.leakyrelu_1(out)

        out = self.maxpool_1(out)

        out = self.conv_2(out)
        out = self.batchnorm_2(out)
        out = self.leakyrelu_2(out)

        out = self.conv_3(out)
        out = self.batchnorm_3(out)
        out = self.leakyrelu_3(out)

        out = self.conv_4(out)
        out = self.batchnorm_4(out)
        out = self.leakyrelu_4(out)

        out = self.conv_5(out)
        out = self.batchnorm_5(out)
        out = self.leakyrelu_5(out)

        out = self.conv_6(out)
        out = self.batchnorm_6(out)
        out = self.leakyrelu_6(out)

        out = self.conv_7(out)
        out = self.batchnorm_7(out)
        out = self.leakyrelu_7(out)

        out = self.conv_8(out)
        out = self.batchnorm_8(out)
        out = self.leakyrelu_8(out)

        out = self.conv_9(out)
        out = self.batchnorm_9(out)
        out = self.leakyrelu_9(out)

        out = self.conv_10(out)
        out = self.batchnorm_10(out)
        out = self.leakyrelu_10(out)

        out = self.conv_11(out)
        out = self.batchnorm_11(out)
        out = self.leakyrelu_11(out)

        out = self.conv_12(out)
        out = self.batchnorm_12(out)
        out = self.leakyrelu_12(out)

        out = self.conv_13(out)
        out = self.batchnorm_13(out)
        out = self.leakyrelu_13(out)

        out = self.conv_14(out)
        out = self.batchnorm_14(out)
        out = self.leakyrelu_14(out)

        out = self.conv_15(out)
        out = self.batchnorm_15(out)
        out = self.leakyrelu_15(out)

        out = self.maxpool_2(out)

        out = self.conv_16(out)
        out = self.batchnorm_16(out)
        out = self.leakyrelu_16(out)

        out = self.conv_17(out)
        out = self.batchnorm_17(out)
        out = self.leakyrelu_17(out)

        out = self.conv_18(out)
        out = self.batchnorm_18(out)
        out = self.leakyrelu_18(out)

        out = self.conv_19(out)
        out = self.batchnorm_19(out)
        out = self.leakyrelu_19(out)

        out = self.maxpool_3(out)

        out = self.conv_20(out)
        out = self.batchnorm_20(out)
        out = self.leakyrelu_20(out)

        out = self.conv_21(out)
        out = self.batchnorm_21(out)
        out = self.leakyrelu_21(out)

        out = self.conv_22(out)
        out = self.batchnorm_22(out)
        out = self.leakyrelu_22(out)

        out = self.conv_23(out)
        out = self.batchnorm_23(out)
        out = self.leakyrelu_23(out)

        # End of darknet
        print("Darknet_Out", out.shape)

        out = self.flatten(out)
        out = self.linear_0(out)
        out = self.dropout(out)
        out = self.leakyrelu_fc(out)
        out = self.linear_1(out)

        # End of model
        print("Model_Out", out.shape)

        return out


def main():

    model = YoloV1()
    input = torch.randn((20, 3, 448, 448))
    model.eval()
    output = model(input)


if __name__ == '__main__':
    main()

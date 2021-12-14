import torch
import torch.nn as nn

from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio
import glob


class Toy(nn.Module):
    def __init__(self):
        super(Toy, self).__init__()

        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm_0 = nn.BatchNorm2d(10)
        self.leakyrelu_0 = nn.LeakyReLU(0.1)

    def forward(self, x):

        out = self.conv_0(x)
        out = self.batchnorm_0(out)
        out = self.leakyrelu_0(out)

        return out


def main():

    model = Toy()

    convert_transform = transforms.ToTensor()
    resize_transform = transforms.Resize((448, 448))

    transform = transforms.Compose([convert_transform, resize_transform])

    image = imageio.imread("data/cat.jpg")
    plt.imshow(image)
    plt.show()

    image = transform(image)

    model.eval()
    image = image[None, :]
    output = model(image)
    print(output.shape)

    output = output[0, ...].permute(1, 2, 0)
    output = output.detach().numpy()
    depth = output.shape[2]
    print(output.shape)

    for i in range(depth):
        fig, ax = plt.subplots(1, figsize=(5, 5))
        ax.set_title(i)
        plt.imshow(output[..., i])
        plt.show()


if __name__ == '__main__':
    main()

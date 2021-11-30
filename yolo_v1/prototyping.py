import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import imageio
import glob


class NNModule(nn.Module):
    def __init__(self, channels):
        super(NNModule, self).__init__()
        self.conv = nn.Conv2d(3, 3, bias=False, kernel_size=4, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(3)
        self.leakyrelu = nn.LeakyReLU(0.9)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(112*112*3, 112*112*3)

    def forward(self, x):
        #x = self.conv(x)
        #x = self.batchnorm(x)
        #x = self.leakyrelu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def plot_batch(batch):
    for image_index in range(batch.shape[0]):
        image = batch[image_index, ...]
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()
        plot_image(image)


def plot_image(image):
    if torch.is_tensor(image):
        image = image.permute(1, 2, 0)  # Permute the axis of the tensor to become an image
        image = image.detach().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.show()


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Deice:", device)

    # Types of preprocessing transforms we want to apply
    convert_transform = transforms.ToTensor()
    resize_transform = transforms.Resize((224, 224))
    jitter_transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    erasing_transform = transforms.RandomErasing()

    transform = transforms.Compose([convert_transform, resize_transform, jitter_transform, erasing_transform])

    list_of_image_tensors = []
    for im_path in glob.glob("data/*.jpg"):
        image = imageio.imread(im_path)
        #plot_image(image)
        image = transform(image)
        #plot_image(image)
        list_of_image_tensors.append(image)

    #image_batch = torch.randn((1, 3, 224, 224))  # Goal is to replace this with actual images
    image_batch = torch.stack(list_of_image_tensors, dim=0)  # (Batch, Channel, Width, Height)

    #plot_batch(image_batch)
    module_ft = NNModule(3).to(device=device)
    image_batch = image_batch.to(device)
    image_batch = module_ft.forward(image_batch)
    image_batch = image_batch.to("cpu")
    image_batch = torch.reshape(image_batch, (5, 3, 112, 112))
    plot_batch(image_batch)


if __name__=="__main__":
    main()
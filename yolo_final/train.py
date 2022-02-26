import config

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import numpy as np

from dataloader import CustomDataset
from utils import draw_y_on_x


def main():

    train_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.transforms, catagory=config.CATEGORY_FILTER)
    val_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.transforms, catagory=config.CATEGORY_FILTER)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    for index, (x, y) in enumerate(train_loader):
        x = draw_y_on_x(x, y)
        grid = torchvision.utils.make_grid(x, nrow=4)
        # Save batch grid as image
        image_dir = "./batch_dir"
        image_dir_exists = os.path.exists(image_dir)
        if not image_dir_exists:
            os.makedirs(image_dir)
        img_name = str(image_dir) + "/batch_" + str(index) + ".png"
        save_image(grid.float() / 255, img_name)

        print(index)
        print(x.shape)
        print(y.shape)

    for index, (x, y) in enumerate(val_loader):
        print(index)
        print(x.shape)
        print(y.shape)


if __name__ == "__main__":
    main()

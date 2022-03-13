import config
import torch
import torch.optim as optim
import torchvision
import time
import os

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model_resnet import YoloV3
from dataloader import CustomDataset
from utils import draw_y_on_x
from utils import draw_yp_on_x
from loss import YoloLoss


def main():

    # Data loading
    train_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.train_transforms, catagory=config.CATEGORY_FILTER)
    val_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.val_transforms, catagory=config.CATEGORY_FILTER)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)

    # Model
    model = YoloV3(num_classes=config.C).to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_function = YoloLoss()
    scalar = torch.cuda.amp.GradScaler()

    # Miscellaneous
    scaled_anchors = (torch.tensor(config.anchors) * torch.tensor(config.Scale).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
    #writer = SummaryWriter()
    current_time = time.time()

    print("Train loader length:", len(train_loader))

    # Training loop
    model.train()
    for cycle, (x, y) in enumerate(train_loader):

        print("Current cycle:", cycle)

        delta_time, current_time = time_function(current_time)
        #writer.add_scalar("Epoch Duration [s]", delta_time, cycle)
        #writer.flush()


def time_function(current_time):
    delta_time = time.time() - current_time
    current_time = time.time()
    return delta_time, current_time


if __name__ == "__main__":
    main()

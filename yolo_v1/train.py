import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import time

from model import Yolov1
from dataset import VOCDataset
from loss import YoloLoss

from utils import intersection_over_union
from utils import non_max_suppression
from utils import mean_average_precision
from utils import cellboxes_to_boxes
from utils import get_bboxes
from utils import plot_image
from utils import save_checkpoint
from utils import load_checkpoint


seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0
EPOCHS = 300
NUM_WORKERS = 1
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "../datasets/voc/images"
LABEL_DIR = "../datasets/voc/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimzer, loss_fn):
    loop = tqdm(train_loader, position=0, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        # Update the progress bar
        loop.set_postfix(loss=loss.item())

    return sum(mean_loss)/len(mean_loss)


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("../datasets/voc/100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset("../datasets/voc/100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    map_list = []
    mean_loss_list = []

    for epoch in range(EPOCHS):

        # for x, y in train_loader:
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #         plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(10)

        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)

        print("Epoch:", epoch)
        print(f"Train mAP: {mean_avg_prec}")
        print(f"Mean loss was {mean_loss}")
        map_list.append(mean_avg_prec)
        mean_loss_list.append(mean_loss)


    plt.plot(map_list)
    plt.show()
    plt.plot(mean_loss_list)
    plt.show()

    for x, y in test_loader:
        x = x.to("cuda")
        for idx in range(BATCH_SIZE):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            imgs = x[idx].permute(1, 2, 0).to("cpu")
            plot_image(imgs, bboxes)


if __name__ == "__main__":
    main()

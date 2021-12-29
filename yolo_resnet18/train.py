import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import time
from torch.utils.tensorboard import SummaryWriter

from loss import YoloLoss
from model import YoloV1
from voc_loader import VOCDataset
from voc_loader import render_batch

from utils import get_bboxes
from utils import mean_average_precision


def main():

    IMG_DIR = "../datasets/voc/images"
    LABEL_DIR = "../datasets/voc/labels"
    MAPPING_FILE = "../datasets/voc/100examples.csv"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 4e-4
    WEIGHT_DECAY = 0.0
    EPOCHS = 100
    LOAD_MODEL_FILE = "retrained_grad_true.pth.tar"
    S = 7
    C = 20
    B = 2
    PRETRAINED = True

    print("Using Device:", DEVICE)

    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, bboxes):
            for t in self.transforms:
                image = t(image)
            return image, bboxes

    # Types of preprocessing transforms we want to apply
    convert_transform = transforms.ToTensor()
    resize_transform = transforms.Resize((448, 448))
    # normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([convert_transform, resize_transform])

    train_dataset = VOCDataset(MAPPING_FILE, S=S, C=C, B=B, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    val_dataset = VOCDataset(MAPPING_FILE, S=S, C=C, B=B, transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    model = YoloV1(S=S, B=B, C=C, pretrained=PRETRAINED).to(DEVICE)

    for counter, param in enumerate(model.parameters()):
        if counter < 63:
            param.requires_grad = True
            print(counter, param.requires_grad)
    model.resnet18.fc.requires_grad = True

    parameter_list = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.Adam(parameter_list, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    yolo_loss = YoloLoss(S=S, C=C, B=B)
    writer = SummaryWriter()
    current_time = time.time()
    mean_loss_list = []

    for epoch in range(EPOCHS):

        mean_loss = []

        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        # Save the model
        if mean_avg_prec > 0.99 or epoch == (EPOCHS - 1):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            print("=> Saving checkpoint")
            torch.save(checkpoint, LOAD_MODEL_FILE)
            time.sleep(10)

        for batch_idx, (x, y) in enumerate(train_loader):

            x, y = x.to(DEVICE), y.to(DEVICE)
            y_p = model(x)
            # y = torch.flatten(y, start_dim=1, end_dim=3)
            y_p = torch.reshape(y_p, (BATCH_SIZE, S, S, C + 5 * B))
            loss = yolo_loss(y_p, y)
            mean_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate average loss
        mean_loss = sum(mean_loss)/len(mean_loss)
        delta_time = time.time() - current_time
        current_time = time.time()

        writer.add_scalar("Average Loss: ", mean_loss, epoch)
        writer.add_scalar("Mean Average Precision: ", mean_avg_prec, epoch)
        writer.add_scalar("Epoch Duration [s]", delta_time, epoch)

        print("Epoch:", epoch)
        mean_loss_list.append(mean_loss)

    writer.close()


if __name__ == '__main__':
    main()

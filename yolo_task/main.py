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
from data_loader import COCODataset
from data_loader import render_batch

from utils import loss_function


def main():

    TRAIN_IMG_DIR = "data/images/train2017/"
    VAL_IMG_DIR = "data/images/val2017/"
    TEST_IMG_DIR = "data/images/test2017/"
    LABEL_DIR = "bbox_labels_train.json"
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.0
    EPOCHS = 50
    TRAIN = False
    LOAD_MODEL_FILE = "epoch.pth.tar"
    S = 7
    C = 100
    B = 2

    print(DEVICE)

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
    transform = Compose([convert_transform, resize_transform])

    train_dataset = COCODataset("bbox_labels_val.json", S=S, C=C, B=B, transform=transform, img_dir=VAL_IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    val_dataset = COCODataset("bbox_labels_val.json", S=S, C=C, B=B, transform=transform, img_dir=VAL_IMG_DIR, label_dir=LABEL_DIR)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=True)

    model = YoloV1(S=S, B=B, C=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    yolo_loss = YoloLoss(S=S, C=C, B=B)
    writer = SummaryWriter()

    # TRAINING
    if TRAIN:

        mean_loss_list = []

        for epoch in range(EPOCHS):

            mean_loss = []

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
            writer.add_scalar("Average Loss: ", mean_loss, epoch)

            print("Epoch:", epoch)
            print(f"Mean loss was {mean_loss}")
            mean_loss_list.append(mean_loss)

            # Save the model
            if True:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                print("=> Saving checkpoint")
                torch.save(checkpoint, LOAD_MODEL_FILE)
                time.sleep(10)

        writer.close()
        plt.plot(mean_loss_list)
        plt.show()

    # VALIDATION
    else:

        print("=> Loading checkpoint")
        model.load_state_dict(torch.load(LOAD_MODEL_FILE)["state_dict"])
        optimizer.load_state_dict(torch.load(LOAD_MODEL_FILE)["optimizer"])

        model.train()  # Sets model to training mode
        model.eval()  # Sets model to evaluation mode
        with torch.no_grad():  # Use with evaluation mode

            for epoch in tqdm(range(1)):
                for i, (x, y) in enumerate(val_loader):
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_p = model(x)
                    x = x.to('cpu')
                    y_p = y_p.to('cpu')
                    y_p = torch.reshape(y_p, (BATCH_SIZE, S, S, C + 5 * B))
                    print("Rendering original labels (y to x)")
                    render_batch(x, y)
                    print("Rendering predicted labels (y_p to x)")
                    render_batch(x, y_p)


if __name__ == '__main__':
    main()

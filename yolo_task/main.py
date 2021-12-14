import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import time

from model import YoloV1
from data_loader import COCODataset


def main():

    TRAIN_IMG_DIR = "data/images/val2017/"
    TEST_IMG_DIR = "data/images/test2017/"
    LABEL_DIR = "bbox_labels_train.json"
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.0
    EPOCHS = 300

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

    train_dataset = COCODataset("bbox_labels_val.json", S=7, C=100, B=2, transform=transform, img_dir=TRAIN_IMG_DIR, label_dir=LABEL_DIR)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True, drop_last=False)

    model = YoloV1(S=7, B=2, C=100).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in tqdm(range(EPOCHS)):

        model.eval()

        for i, (image_batch, label_batch) in enumerate(train_loader):

            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)

            # with torch.no_grad():
            output = model(image_batch)
            # print("Output shape", output.shape)
            # print("Label shape", label_batch.shape)




if __name__ == '__main__':
    main()

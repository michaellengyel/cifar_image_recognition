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
from utils import boxes_from_y
from utils import boxes_from_yp
from utils import save_checkpoint
from utils import load_checkpoint
from utils import denormalize

from loss import YoloLoss

# Metrics import
from torchmetrics import MeanMetric
from torchvision.ops import box_convert
import sklearn.metrics  # For classification
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score

# Accuracy
# Precision
# Recall
# F-measure
# Confusion matrix
# ROC curve


def main():

    # Data loading
    # train_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.train_transforms, catagory=config.CATEGORY_FILTER)
    val_dataset = CustomDataset(root=config.root_val, annFile=config.annFile_val, transforms=config.val_transforms, catagory=config.CATEGORY_FILTER)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)


    # Model
    model = YoloV3(num_classes=config.C).to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # Miscellaneous
    scaled_anchors = (torch.tensor(config.anchors) * torch.tensor(config.Scale).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)

    # Loading previously saved model weights
    load_checkpoint("res50_35k.pth.tar", model, optimizer, config.LEARNING_RATE)

    # Rendering loop
    model.eval()
    for cycle, (x, y) in enumerate(val_loader):
        with torch.no_grad():
            x_gpu = x.to(config.DEVICE)
            yp = model(x_gpu)
            yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]
        x = denormalize(x)*255
        draw_y_on_x(x, y)
        draw_yp_on_x(x, yp, probability_threshold=0.5, anchors=config.anchors)
        # Save batch grid as image
        image_dir = "./batch_dir"
        image_dir_exists = os.path.exists(image_dir)
        if not image_dir_exists:
            os.makedirs(image_dir)
        img_name = str(image_dir) + "/batch_" + str(cycle) + ".png"
        save_image(x/255, img_name)


if __name__ == "__main__":
    main()

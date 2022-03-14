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
    train_dataset = CustomDataset(root=config.root_train, annFile=config.annFile_train, transforms=config.train_transforms, catagory=config.CATEGORY_FILTER)
    val_dataset = CustomDataset(root=config.root_val, annFile=config.annFile_val, transforms=config.val_transforms, catagory=config.CATEGORY_FILTER)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=8, num_workers=2, pin_memory=True, shuffle=False, drop_last=True)

    # Model
    model = YoloV3(num_classes=config.C).to(device=config.DEVICE)
    # from model_external import YOLOv3
    # model = YOLOv3(num_classes=90).to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    from loss_external import YoloLoss
    loss_function = YoloLoss().to(device=config.DEVICE)

    # Miscellaneous
    scaled_anchors = (torch.tensor(config.anchors) * torch.tensor(config.Scale).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
    writer = SummaryWriter()
    current_time = time.time()

    # Loading previously saved model weights
    if config.LOAD_MODEL:
        load_checkpoint("cp.pth.tar", model, optimizer, config.LEARNING_RATE)

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # Training loop
    for cycle in range(config.CYCLES):

        print("Cycle:", cycle)

        x, y = next(iter(val_loader))
        x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))
        yp = model(x)
        loss_0 = loss_function(predictions=yp[0], target=y0, anchors=scaled_anchors[0])
        loss_1 = loss_function(predictions=yp[1], target=y1, anchors=scaled_anchors[1])
        loss_2 = loss_function(predictions=yp[2], target=y2, anchors=scaled_anchors[2])
        loss = loss_0 + loss_1 + loss_2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Run validation
        if cycle % 100 == 0 and cycle != 0:
            model.eval()
            losses = []
            with torch.no_grad():
                x, y = next(iter(val_loader))
                x = x.to(config.DEVICE)
                y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))
                yp = model(x)
                loss_0 = loss_function(predictions=yp[0], target=y0, anchors=scaled_anchors[0])
                loss_1 = loss_function(predictions=yp[1], target=y1, anchors=scaled_anchors[1])
                loss_2 = loss_function(predictions=yp[2], target=y2, anchors=scaled_anchors[2])
                loss = loss_0 + loss_1 + loss_2
                losses.append(loss)
            avg_val_loss = sum(losses) / len(losses)
            writer.add_scalar("val_loss: ", avg_val_loss, cycle)
            model.train()

        # Run validation
        """
        if cycle % 100 == 0 and cycle != 0:
            model.eval()
            x, y = next(iter(val_loader))
            x = x.float()
            x = x.to(config.DEVICE)
            # y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))
            with torch.no_grad():
                yp = model(x)
                # Move predictions to cpu
                yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]
                # boxes_from_yp(yp) returns all yp bboxes in a batch
                yp_boxes = boxes_from_yp(yp=yp, iou_threshold=config.MAP_IOU_THRESH, threshold=config.CONF_THRESHOLD)
                # boxes_from_y(y) returns all y bboxes in a batch
                y_boxes = boxes_from_y(y=y)
        """

        # Save model
        if cycle % 1000 == 0 and cycle != 0:
            save_checkpoint(model, optimizer, cycle, filename=config.CHECKPOINT_FILE)

        # Rendering loop
        if cycle % 100 == 0 and cycle != 0:
            model.eval()
            x, y = next(iter(val_loader))
            with torch.no_grad():
                x_gpu = x.to(config.DEVICE)
                yp = model(x_gpu)
                yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]
            x = denormalize(x) * 255
            draw_y_on_x(x, y)
            draw_yp_on_x(x, yp, probability_threshold=0.5, anchors=config.anchors)
            # Save batch grid as image
            image_dir = "./batch_dir"
            image_dir_exists = os.path.exists(image_dir)
            if not image_dir_exists:
                os.makedirs(image_dir)
            img_name = str(image_dir) + "/batch_" + str(cycle) + ".png"
            save_image(x / 255, img_name)
            model.train()

        writer.add_scalar("train_loss: ", loss.item(), cycle)
        delta_time, current_time = time_function(current_time)
        writer.add_scalar("Epoch Duration [s]", delta_time, cycle)
        writer.flush()


def time_function(current_time):
    delta_time = time.time() - current_time
    current_time = time.time()
    return delta_time, current_time


if __name__ == "__main__":
    main()

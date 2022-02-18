import config
import torch
import torch.optim as optim
import torchvision
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from model_resnet import YoloV3
from dataloader import YoloDataset
from loss import YoloLoss

from utils import draw_y_on_x
from utils import draw_yp_on_x
from utils import time_function
from utils import save_checkpoint
from utils import load_checkpoint
from utils import boxes_from_yp
from utils import boxes_from_y
from utils import calc_batch_precision_recall


def main():

    # Data loading
    train_csv_path = config.DATASET + "100examples.csv"
    test_csv_path = config.DATASET + "100examples.csv"
    train_dataset = YoloDataset(train_csv_path, transforms=config.max_transforms, Scale=config.Scale, image_dir=config.IMG_DIR, label_dir=config.LABEL_DIR, anchor_boxes=config.ANCHORS, number_of_anchors=3, number_of_scales=3, ignore_iou_threshold=0.5, num_anchors_per_scale=3)
    test_dataset = YoloDataset(test_csv_path, transforms=config.max_transforms, Scale=config.Scale, image_dir=config.IMG_DIR, label_dir=config.LABEL_DIR, anchor_boxes=config.ANCHORS, number_of_anchors=3, number_of_scales=3, ignore_iou_threshold=0.5, num_anchors_per_scale=3)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=False, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=False, drop_last=True)

    # Model
    model = YoloV3(num_classes=config.C).to(device=config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_function = YoloLoss()
    scalar = torch.cuda.amp.GradScaler()

    # Miscellaneous
    scaled_anchors = (torch.tensor(config.ANCHORS) * torch.tensor(config.Scale).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)).to(config.DEVICE)
    writer = SummaryWriter()
    current_time = time.time()

    # Loading previously saved model weights
    if config.LOAD_MODEL:
        load_checkpoint(config.TRAINED_FILE, model, optimizer, config.LEARNING_RATE)

    # Epoch loop
    for epoch in range(config.NUM_EPOCHS):

        print("Epoch:", epoch)

        # Run training
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.float()
            x = x.to(config.DEVICE)
            y0, y1, y2 = (y[0].to(config.DEVICE), y[1].to(config.DEVICE), y[2].to(config.DEVICE))
            with torch.cuda.amp.autocast():
                y_p = model(x)
                loss_0 = loss_function(predictions=y_p[0], target=y0, anchors=scaled_anchors[0])
                loss_1 = loss_function(predictions=y_p[1], target=y1, anchors=scaled_anchors[1])
                loss_2 = loss_function(predictions=y_p[2], target=y2, anchors=scaled_anchors[2])
                loss = loss_0 + loss_1 + loss_2
                losses.append(loss.item())
                optimizer.zero_grad()
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()
        mean_loss = sum(losses) / len(losses)
        writer.add_scalar("loss: ", mean_loss, epoch)

        # Run evaluation
        if config.SAVE_MODEL and epoch > 0 and epoch % config.SAVE_FREQUENCY == 0:

            batch_recalls, batch_precisions = [], []

            model.eval()
            for x, y in test_loader:
                x = x.float()
                x = x.to(config.DEVICE)
                with torch.no_grad():
                    yp = model(x)

                # Move predictions to cpu
                yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]

                # boxes_from_yp(yp) returns all yp bboxes in a batch
                yp_boxes = boxes_from_yp(yp=yp, iou_threshold=config.MAP_IOU_THRESH, threshold=config.CONF_THRESHOLD)

                # boxes_from_y(y) returns all y bboxes in a batch
                y_boxes = boxes_from_y(y=y)

                # Calculate precision for batch
                precision, recall = calc_batch_precision_recall(y_boxes=y_boxes, yp_boxes=yp_boxes, iou_threshold=config.MAP_IOU_THRESH)

                batch_precisions.append(precision)
                batch_recalls.append(recall)

            mean_precision = sum(batch_precisions) / len(batch_precisions)
            mean_recall = sum(batch_recalls) / len(batch_recalls)

            # Log evaluation variables
            writer.add_scalar("precision", mean_precision, epoch)
            writer.add_scalar("recall", mean_recall, epoch)

        if config.RENDER and epoch > 0 and epoch % config.SAVE_FREQUENCY == 0:
            # Render prediction bboxes
            x, y = next(x for i, x in enumerate(test_loader) if i == 5)
            model.eval()
            with torch.no_grad():
                x_gpu = x.float()
                x_gpu = x_gpu.to(config.DEVICE)
                yp = model(x_gpu)
                yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]
            x = draw_yp_on_x(x, yp, probability_threshold=0.5, anchors=config.ANCHORS)
            x = draw_y_on_x(x, y)
            grid = torchvision.utils.make_grid(x, nrow=4)
            writer.add_image("yp on x", grid, global_step=epoch)

        # Save model
        if config.SAVE_MODEL and epoch > 0 and epoch % config.SAVE_FREQUENCY == 0:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)

        delta_time, current_time = time_function(current_time)
        writer.add_scalar("Epoch Duration [s]", delta_time, epoch)

        writer.flush()


if __name__ == "__main__":
    main()

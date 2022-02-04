import config
import torch
import torch.optim as optim
import torchvision
import time

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from model_resnet import YoloV3
from dataloader import YoloDataset
from loss import YoloLoss
from torchvision_utils import draw_bounding_boxes

from utils import time_function
from utils import check_class_accuracy
from utils import get_evaluation_bboxes
from utils import mean_average_precision
from utils import save_checkpoint
from utils import load_checkpoint
from utils import draw_y_on_x
from utils import draw_yp_on_x


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

        # Run evaluation
        if config.SAVE_MODEL and epoch > 0 and epoch % config.SAVE_FREQUENCY == 0:

            # Perform evaluation calculations
            model.eval()
            class_accuracy, no_object_accuracy, obj_accuracy = check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            prediction_boxes, true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=config.NMS_IOU_THRESH, anchors=config.ANCHORS, threshold=config.CONF_THRESHOLD)
            mean_avg_prec = mean_average_precision(prediction_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESH, box_format='midpoint', num_classes=config.C)

            # Log evaluation variables
            writer.add_scalar("Average Loss: ", mean_loss, epoch)
            writer.add_scalar("class_accuracy", class_accuracy, epoch)
            writer.add_scalar("no_object_accuracy", no_object_accuracy, epoch)
            writer.add_scalar("obj_accuracy", obj_accuracy, epoch)
            writer.add_scalar("mean_avg_prec", mean_avg_prec, epoch)

            # Render prediction bboxes
            x, y = next(x for i, x in enumerate(test_loader) if i == 5)
            model.eval()
            with torch.no_grad():
                x_gpu = x.float()
                x_gpu = x_gpu.to(config.DEVICE)
                yp = model(x_gpu)
                yp = [yp[0].to('cpu'), yp[1].to('cpu'), yp[2].to('cpu')]
            x = draw_yp_on_x(x, yp, probability_threshold=0.5, anchors=config.ANCHORS)
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


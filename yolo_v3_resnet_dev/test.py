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

from utils_depreciated import time_function
from utils_depreciated import check_class_accuracy
from utils_depreciated import get_evaluation_bboxes
from utils_depreciated import mean_average_precision
from utils_depreciated import get_evaluation_bboxes


def main():

    # Data loading
    test_csv_path = config.DATASET + "100examples.csv"
    test_dataset = YoloDataset(test_csv_path, transforms=config.max_transforms, Scale=config.Scale, image_dir=config.IMG_DIR, label_dir=config.LABEL_DIR, anchor_boxes=config.ANCHORS, number_of_anchors=3, number_of_scales=3, ignore_iou_threshold=0.5, num_anchors_per_scale=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=False, drop_last=True)

    # Model
    model = YoloV3(num_classes=config.C)
    optimizer = optim.Adam(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loss_function = YoloLoss()
    scalar = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    model.to(device=config.DEVICE)
    model.eval()

    def load_checkpoint(checkpoint_file, model, optimizer, lr):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    # all_pred_boxes, all_true_boxes = get_evaluation_bboxes(test_loader, model, iou_threshold=0.5, anchors=config.ANCHORS, threshold=0.5, box_format="midpoint", device="cuda")

    model.eval()
    train_idx = 0
    anchors = config.ANCHORS
    threshold = 0.9
    from utils_depreciated import cells_to_bboxes
    from utils_depreciated import non_max_suppression
    for batch_idx, (x, labels) in enumerate(test_loader):
        x = x.float()
        x = x.to("cuda")

        all_pred_boxes = []
        all_true_boxes = []

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to("cuda") * S
            boxes_scale_i = cells_to_bboxes(predictions[i], anchor, S=S, is_preds=True)
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=0.9, threshold=0.9, box_format="midpoint")

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

        from torchvision_utils import draw_bounding_boxes

        x = x.to('cpu') * 255
        #boxes = all_pred_boxes #cells_to_boxes_render(y)
        boxes = [(torch.tensor(x)[3:] * torch.tensor([416, 416, 416, 416])).tolist() for x in all_pred_boxes]
        boxes = torch.tensor(boxes)
        for idx in range(x.shape[0]):
            x[idx, ...] = draw_bounding_boxes(image=x[idx, ...].type(torch.uint8), boxes=boxes, colors=None, width=2, pad_value=2)
        grid = torchvision.utils.make_grid(x)
        writer.add_image("images", grid, global_step=batch_idx)

        writer.flush()

    """
    for x, y in test_loader:
        x = x.float()
        x = x.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            y_p = model(x)
            print(y_p[0].shape)
            print(y_p[1].shape)
            print(y_p[2].shape)

            print(y_p[0][0].shape)
            print("d")
    """


if __name__ == "__main__":
    main()

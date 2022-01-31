import config
import torch
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont, ImageColor

from torchvision_utils import draw_bounding_boxes


def anchors_to_corners(anchors, image_size):
    boxes = []
    for i in range(anchors.shape[0]):
        half_image_size = image_size / 2
        boxes.append([half_image_size - anchors[i, 0] * image_size / 2,
                      half_image_size - anchors[i, 1] * image_size / 2,
                      half_image_size + anchors[i, 0] * image_size / 2,
                      half_image_size + anchors[i, 1] * image_size / 2])
    return torch.tensor(boxes)


def main():

    image_size = 416
    anchors = torch.tensor(config.ANCHORS[0] + config.ANCHORS[1] + config.ANCHORS[2])
    x = torch.rand((3, image_size, image_size))

    # boxes = 416 * torch.rand((12, 4))
    boxes = anchors_to_corners(anchors, image_size)

    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    image = draw_bounding_boxes(image=x.type(torch.uint8), boxes=boxes, colors=(255, 255, 255), labels=labels)
    image = image.permute(1, 2, 0)

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()

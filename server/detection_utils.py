from matplotlib import patches
import torchvision
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw as D


def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    y, x = img.shape[0], img.shape[1]
    x_scalar = 224 / x
    y_scalar = 224 / y
    for box in (target['boxes']):
        x, y, width, height = box[0] / x_scalar, box[1] / y_scalar, (box[2] - box[0]) / x_scalar, (
                box[3] - box[1]) / y_scalar
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


def get_cropped_images(img, target):
    images = []
    positions = []
    y, x = img.shape[0], img.shape[1]
    x_scalar = 224 / x
    y_scalar = 224 / y
    img2 = img.copy()
    img2 = torchvision.transforms.ToPILImage()(img2)
    draw = D.Draw(img2)
    for box in (target['boxes']):
        # x, y, width, height  = box[0]/x_scalar, box[1]/y_scalar, (box[2]-box[0])/x_scalar, (box[3]-box[1])/y_scalar
        # im = img.crop((bxs[i][0]/x_scalar,bxs[i][1]/y_scalar,bxs[i][2]/x_scalar,bxs[i][3]/y_scalar))
        y1 = int(box[1] / y_scalar)
        y2 = int(box[3] / y_scalar)
        x1 = int(box[0] / x_scalar)
        x2 = int(box[2] / x_scalar)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red" , width= 5)
        im = img[y1:y2, x1:x2]
        px = x1 + (x2 - x1) / 2
        py = y1 + (y2 - y1) / 2
        p = get_position(py, px, y, x)
        images.append(im)
        positions.append(p)
    return images, positions, img2


def get_position(py, px, img_height, img_width):
    y = img_height / 3
    x = img_width / 3

    if px < x and py < y:
        position = "far left"
    elif 2 * x > px > x and py < y:
        position = "far center"
    elif px > 2 * x and py < y:
        position = "far right"
    elif px < x and 2 * y > py > y:
        position = "mid left"
    elif 2 * x > px > x and 2 * y > py > y:
        position = "mid center"
    elif px > 2 * x and 2 * y > py > y:
        position = "mid right"
    elif px < x and py > 2 * y:
        position = "near left"
    elif 2 * x > px > x and py > 2 * y:
        position = "near center"
    else:
        position = "near right"
    return position

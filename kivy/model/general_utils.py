
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
import os
import numpy as np
import cv2
from tqdm import tqdm
import PIL.Image
from master.kivy.model.detection_utils import *
from  master.kivy.model.classification_utils import *


## Detection MODEL
def detection_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 100
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)
    path = 'D:/College/Semester 9/GP/Codes/master/detection weights/fasterrcnn_uec26.pth'
    # if torch.cuda.is_available():
    #     model.load_state_dict(torch.load(path))
    # else:
    #     model.load_state_dict(torch.load(path, map_location=device))
    model.load_state_dict(torch.load(path, map_location=device))

    model.eval()  # put the model in evaluation mode
    return model


##

def detect(model, img):
    image = torchvision.transforms.ToPILImage()(img)
    transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, ], [0.5, ])])
    image = transform(image)
    ##

    ## Predict Detection Images
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    # print('MODEL OUTPUT\n')

    nms_prediction = apply_nms(prediction, iou_thresh=0.08)
    # print(nms_prediction)
    images, plates_positions,bb_img = get_cropped_images(img, nms_prediction)
    ##

    ## Show Cropped Images
    # plot_img_bbox(img, nms_prediction)
    return images, plates_positions, bb_img


def classification_model():
    K.clear_session()
    model_best = load_model("D:/College/Semester 9/GP/Codes/master/classification "
                            "weights/54_weights/weights_2/weights-improvement-13-0.85.hdf5", compile=False)
    return model_best


def classify(model_best, images, food_list):
    # for i in images:
    #     f.add_subplot(1, len(images), c + 1)
    #     c += 1
    #     plt.imshow(i)
    #     plt.show()

    with tf.device('/device:CPU:0'):
        predicted_labels = predict_class(model_best, images, food_list, False)
    return predicted_labels

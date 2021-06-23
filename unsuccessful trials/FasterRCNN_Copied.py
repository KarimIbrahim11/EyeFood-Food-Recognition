import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import PIL.Image


# import parallelTestModule

def loadPascal():
    images, targets = [], []
    boxes, labels, image_id, iscrowd = [], [], [], []
    directory = 'D:/College/Semester 9/GP/Codes/pascal2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            try:
                tree = ET.parse(directory + filename)
                root = tree.getroot()
                key = os.path.splitext(filename)[0]

                imgfilename = os.path.join(
                    'D:/College/Semester 9/GP/Codes/pascal2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
                    '/JPEGImages/',
                    key + '.jpg')
                img = cv2.imread(imgfilename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                y, x = img.shape[0], img.shape[1]
                x_scalar = 224 / x
                y_scalar = 224 / y

                info = {}
                for bndbox in root.iter('bndbox'):
                    info[key] = []

                    info[key].append(float(bndbox[0].text))
                    info[key].append(float(bndbox[1].text))
                    info[key].append(float(bndbox[2].text))
                    info[key].append(float(bndbox[3].text))

                    target = {}

                    x1 = int(info[key][0] * x_scalar)
                    y1 = int(info[key][1] * y_scalar)
                    x2 = int(info[key][2] * x_scalar)
                    y2 = int(info[key][3] * y_scalar)
                    poly = [x1, y1, x2, y2]
                    area = (poly[0] - poly[2]) * (poly[1] - poly[3])
                    poly = torch.tensor(poly)
                    poly = torch.unsqueeze(poly, 0)

                    target['boxes'] = poly
                    target['labels'] = torch.tensor([0])  # 0 means non food
                    target['image_id'] = torch.tensor([int(key)])
                    target['area'] = torch.tensor([area])
                    target['iscrowd'] = torch.tensor([0])

                    images.append(img)
                    targets.append(target)

            except FileNotFoundError:
                print('cannot find file ', filename)

    return images, targets


def LoadFoodData(folder_path, n_cls):
    targets = []
    images = []
    boxes, labels, image_id, iscrowd = [], [], [], []
    for i in tqdm(range(1, n_cls + 1)):
        info = {}
        path = os.path.join(folder_path, str(i))
        file = open(path + '/bb_info.txt')
        txt = file.read()
        file.close()
        txt = txt.split('\n')
        # Making a dict of text file
        for j in txt[1:]:
            if len(j) > 0:
                temp = j.split(' ')
                info[temp[0]] = [int(x) for x in temp[1:]]
        # For loading images and targets
        for key in info:
            target = {}
            filename = os.path.join(path, key + '.jpg')
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            y, x = img.shape[0], img.shape[1]

            x_scalar = 224 / x
            y_scalar = 224 / y
            x1 = int(info[key][0] * x_scalar)
            y1 = int(info[key][1] * y_scalar)
            x2 = int(info[key][2] * x_scalar)
            y2 = int(info[key][3] * y_scalar)
            poly = [x1, y1, x2, y2]
            area = (poly[0] - poly[2]) * (poly[1] - poly[3])
            poly = torch.tensor(poly)
            poly = torch.unsqueeze(poly, 0)

            target['boxes'] = poly
            target['labels'] = torch.tensor([1])  # replce i with 1
            target['image_id'] = torch.tensor([int(key)])
            target['area'] = torch.tensor([area])
            target['iscrowd'] = torch.tensor([0])

            images.append(img)
            targets.append(target)

    return images, targets


class FoodData(Dataset):
    def __init__(self, images, targets, transforms=None):
        self.images = images
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = torchvision.transforms.ToPILImage()(image)
        if self.transforms:
            image = self.transforms(image)
        return image, target


def collate(batch):
    return tuple(zip(*batch))


def predict(model, test_image_name):
    test_image = PIL.Image.open(test_image_name).convert('RGB')
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    print(test_image_tensor)
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        #         ps = torch.exp(out)
        #         print(ps)
        #         topk, topclass = ps.topk(1, dim=1)
        #         print(topclass.cpu().numpy()[0][0])
        print(out)
        return out, test_image_tensor


if __name__ == '__main__':
    ##
    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)

    # testing
    # import matplotlib
    # from matplotlib import pyplot, patches

    # img = matplotlib.image.imread('../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages
    # /000005.jpg')

    # figure, ax = pyplot.subplots(1)
    # ax.imshow(img)

    # print(os.path.splitext('0005.txt')[0])

    # tree = ET.parse('../input/pascal-voc-2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/000005.xml')
    # root = tree.getroot()

    # for bndbox in root.iter('bndbox'):
    #     print(bndbox[0].text)

    # print(root[6][4][0].text)

    # for child in root:
    #     if child.tag == 'object':
    #         for gchild in child:
    #             if gchild.tag == 'bndbox':
    #                 print(gchild[0].text)
    ##

    im, tar = loadPascal()

    print("Khalasna pascal ")

    images, targets = LoadFoodData('D:/College/Semester 9/GP/Codes/UECFOOD100/', 50)

    images = images + im
    targets = targets + tar

    train_images, test_images, train_targets, test_targets = train_test_split(images, targets, test_size=0.2,
                                                                              random_state=7)

    transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, ], [0.5, ])])
    traindata = FoodData(train_images, train_targets, transform)
    trainloader = DataLoader(traindata, batch_size=8, shuffle=True, collate_fn=collate, num_workers=4)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2  # 100
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=3e-4)

    import warnings

    warnings.filterwarnings('ignore')

    epochs = 4
    iteration = 100
    itr = 0
    loss = 0
    for e in range(epochs):
        print('Epoch {}'.format(e + 1))
        for img, tar in trainloader:
            img = list(image.to(device) for image in img)
            tar = [{k: v.to(device) for k, v in t.items()} for t in tar]

            optimizer.zero_grad()
            loss_dict = model(img, tar)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            losses.backward()
            optimizer.step()

            loss += loss_value
            itr += 1
            if itr % iteration == 0:
                print('Iteration:{}\tLoss:{}'.format(itr, (loss / iteration)))
                loss = 0

    torch.save(model.state_dict(), 'fasterrcnn_foodtracker.pth')

    path = './fasterrcnn_foodtracker.pth'
    model.load_state_dict(torch.load(path))
    model.eval()

    from torch.autograd import Variable
    import PIL.Image

    out, test_image_tensor = predict(model, './UECFOOD100/1/13723.jpg')

    print(test_image_tensor.cpu().numpy().shape)

    import matplotlib
    from matplotlib import pyplot, patches

    img = matplotlib.image.imread("./UECFOOD100/1/13723.jpg")

    figure, ax = pyplot.subplots(1)

    bxs = out[0]['boxes'].cpu().numpy()
    print(bxs)

    cbxs = len(bxs)
    for i in range(cbxs):
        rect = patches.Rectangle((bxs[i][0], bxs[i][1]), bxs[i][2], bxs[i][3], edgecolor='r', facecolor="none")
        ax.imshow(img)
        ax.add_patch(rect)

        im = PIL.Image.open("./UECFOOD100/1/13723.jpg")
        newsize = (224, 224)
        im = im.resize(newsize, PIL.Image.ANTIALIAS)
        # Shows the image in image viewer
        figure, ax = pyplot.subplots(1)
        for i in range(cbxs):
            rect = patches.Rectangle((bxs[i][0], bxs[i][1]), bxs[i][2], bxs[i][3], edgecolor='r', facecolor="none")
            ax.imshow(im)
            ax.add_patch(rect)

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import PIL.Image

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 100
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
device = torch.device('cpu')
model.to(device)

transform = torchvision.transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5,], [0.5,])])

path = '/home/amir/PycharmProjects/Food-Recognition/fasterrcnn_foodtracker.pth'
model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
model.eval()


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
        return out


out = predict(model, 'twodishes.jpg')

import matplotlib
from matplotlib import pyplot, patches

img = matplotlib.image.imread("twodishes.jpg")

figure, ax = pyplot.subplots(1)

bxs = out[0]['boxes'].cpu().numpy()
print(bxs)

cbxs = len(bxs)
for i in range(cbxs):
    rect = patches.Rectangle((7*bxs[i][0],5*bxs[i][1]), 7*bxs[i][2], 6*bxs[i][3], edgecolor='r', facecolor="none")
    ax.imshow(img)
    ax.add_patch(rect)


im = PIL.Image.open("twodishes.jpg")
newsize = (224, 224)
im = im.resize(newsize, PIL.Image.ANTIALIAS)
# Shows the image in image viewer
figure, ax = pyplot.subplots(1)
for i in range(cbxs):
    rect = patches.Rectangle((bxs[i][0],bxs[i][1]), bxs[i][2], bxs[i][3], edgecolor='r', facecolor="none")
    ax.imshow(im)
    ax.add_patch(rect)


# im1 = im.crop((bxs[0][0], bxs[0][1], bxs[0][2], bxs[0][3]))
# im1.show()
for i in range(cbxs):
    im1 = im.crop((bxs[i][0], bxs[i][1], bxs[i][2], bxs[i][3]))
    im1.show()

pyplot.show()

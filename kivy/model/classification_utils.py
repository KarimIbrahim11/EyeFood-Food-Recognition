from tensorflow.keras.preprocessing import image
import numpy as np

import cv2


def predict_class(model, images, labels, show=True):
    dim = (299, 299)
    predicted_labels = []
    for img in images:
        img = cv2.resize(img, dim)
        # img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        index = np.argmax(pred)
        labels.sort()
        pred_value = labels[index]
        predicted_labels.append(pred_value)
        # if show:
        #     plt.imshow(img[0])
        #     plt.axis('off')
        #     plt.title(pred_value)
        #     plt.show()

    return predicted_labels

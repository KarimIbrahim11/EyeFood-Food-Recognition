from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import cv2


def predict_class(interpreter, images, labels, show=True):
    dim = (299, 299)
    predicted_labels = []
    for img in images:
        img = cv2.resize(img, dim)
        # img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        # # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("input details", input_details)
        print("output details", output_details)

        input_shape = input_details[0]['shape']
        print(input_shape)
        print(img.shape)
        #
        # height = input_details[0]['shape'][1]
        # width = input_details[0]['shape'][2]
        #
        # try:
        #     img = cv2.resize(img, (width, height))
        # except Exception as e:
        #     print(str(e))

        input_data = np.array(img, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

        labels.sort()
        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]
        for i in top_k:
            print('{:08.6f}: {}'.format(float(results[i]), labels[i]))

        # pred = model.predict(img)
        index = np.argmax(results)

        pred_value = labels[index]
        predicted_labels.append(pred_value)
        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()

    return predicted_labels

from PIL.Image import Image
import os
from master.server.general_utils import *
from flask import Flask, request, jsonify, send_file

fileReader = open('labels.txt', 'r')
food_list = [line.rstrip() for line in fileReader.readlines()]
fileReader.close()
model1 = detection_model()
model2 = classification_model()

app = Flask(__name__)


@app.route("/test", methods=["POST"])
def process_image():
    print(request.files)
    if os.path.exists("bb_img.jpg"):
        os.remove("bb_img.jpg")
    file = request.files["image"]
    # Read the image via file.stream
    # print(file)
    # file.save('test.jpg')
    img = PIL.Image.open(file).convert('RGB')
    # img.show()
    img = np.array(img)
    images, plates_positions, _ = detect(model1, img)
    labels = classify(model2, images, food_list)
    print(labels, plates_positions)

    lp = ""
    for l, p in zip(labels, plates_positions):
        lp += l + " in " + p + ", "
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # files = []
    # imencoded = cv2.imencode(".jpg", _)[1]
    # file = {'file': ('image.jpg', imencoded.tostring(), 'image/jpeg')}
    # files.append(file)
    _.save("bb_img.jpg")
    # _.show()
    response = send_file('bb_img.jpg', 'image/jpeg')
    response.headers['lp'] = lp
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)

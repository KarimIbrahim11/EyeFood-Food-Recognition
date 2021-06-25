import os

from PIL import Image

path = 'D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/Koshary/koshari_t/'
newpath = 'D:/College/Semester 9/GP/Codes/Datasets/Custom Dataset/Koshary/'


def resize():
    c = "Mahshi"
    i = 1
    for item in os.listdir(path):
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            im = im.convert('RGB')
            w, h = im.size
            if w > 950 and h > 950:
                w = w // 3
                h = h // 3
                im = im.resize((w, h), Image.ANTIALIAS)
            elif w > 600 and h > 600:
                w = w // 2
                h = h // 2
                im = im.resize((w, h), Image.ANTIALIAS)
            im.save(newpath + c + str(i) + '.jpg')
            i += 1

            # imResize = im.resize((200,200), Image.ANTIALIAS)
            # imResize.save(f + ' resized.jpg', 'JPEG', quality=90)


resize()

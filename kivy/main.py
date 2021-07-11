import PIL
import cv2
from kivy.app import App
from kivy.core.image import Texture
from kivy.properties import StringProperty
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.carousel import Carousel
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager, Screen
from model.general_utils import *

from kivy.utils import platform

if platform == "android":
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])

temp = 0
images = list()
plates_positions = list()
labels = list()
bb_img = 0

popup = 0


#
# FileChooserIconView:
#             id: filechooser
#             on_submit : root.selected(filechooser.selection)


class F(FileChooserIconView):
    def on_submit(self, selected, touch=None):
        extensions = ['jpg', 'png', 'jpeg']
        name = selected[0]
        splits = name.split('.')
        extension = splits[-1].lower()

        # To ensure it is an image
        if extension in extensions:
            model_run(name)
            App.get_running_app().root.current = "third"
            popup.dismiss()


class CheckScreen(Screen):
    def show_popup(self):
        global popup
        fileViewer = F()
        popup = Popup(title="Choose an image", content=fileViewer, size_hint=(0.8, 0.8))
        popup.open()


class CameraClickScreen(Screen):
    def on_enter(self, *args):
        camera = self.ids['camera']
        camera.play = True

    def capture(self):
        camera = self.ids['camera']
        camera.play = False
        camera.export_as_image().save('test' + str(temp) + '.jpg')
        model_run('test' + str(temp) + '.jpg')
        self.parent.current = "third"
        self.parent.transition.direction = "left"


class ResultScreen(Screen):
    # path = StringProperty('test2.jpg')
    # def on_enter(self, *args):
    #     global temp
    #     image = self.ids['image']
    #     image.source = 'test' + str(temp) + '.jpg'
    #     temp = 1 - temp
    #
    # def on_leave(self, *args):
    #     image = self.ids['image']
    #     image.source = ''

    def on_enter(self, *args):
        carousel = self.ids['carousel']
        carousel.clear_widgets()
        counter = 0

        # Detection Images

        layout = FloatLayout()
        layout.size_hint = (1, 1)
        im = Image()
        im.allow_stretch = True
        im.keep_ratio = True
        im.size_hint = (0.8, 0.8)
        im.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        i = np.array(bb_img)
        i = i[:, :, ::-1].copy()
        i1 = cv2.flip(i, 0)
        i2 = i1.tostring()
        image_texture = Texture.create(size=(i.shape[1], i.shape[0]))
        image_texture.blit_buffer(i2, colorfmt='bgr', bufferfmt='ubyte')
        im.texture = image_texture
        lbl = Label(text="Detection Module Output",font_size = 30,color = (0,0,0,1))
        lbl.size_hint = (None,None)
        lbl.pos_hint = {'center_x': 0.5, 'y': 0.1}
        lbl.size = lbl.texture_size
        layout.add_widget(im)
        layout.add_widget(lbl)
        carousel.add_widget(layout)

        # Classification images
        for i in images:
            layout = FloatLayout()
            im = Image()
            im.allow_stretch = True
            im.keep_ratio = True
            im.size_hint = (0.8, 0.8)
            im.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
            i1 = cv2.flip(i, 0)
            i2 = i1.tostring()
            image_texture = Texture.create(size=(i.shape[1], i.shape[0]))
            image_texture.blit_buffer(i2, colorfmt='rgb', bufferfmt='ubyte')
            im.texture = image_texture
            lbl = Label(text=labels[counter] + " in the " + plates_positions[counter],font_size = 30, color = (0,0,0,1))
            lbl.size_hint = (None, None)
            lbl.pos_hint = {'center_x': 0.5, 'y': 0.1}
            lbl.size = lbl.texture_size
            layout.add_widget(im)
            layout.add_widget(lbl)
            carousel.add_widget(layout)
            counter += 1


class WindowManager(ScreenManager):
    pass


GUI = Builder.load_file('my.kv')


class TestCamera(App):

    def build(self):
        return GUI


fileReader = open('model/labels.txt', 'r')
food_list = [line.rstrip() for line in fileReader.readlines()]
fileReader.close()
model1 = detection_model()
model2 = classification_model()


def model_run(img_path):
    global images, plates_positions, labels, bb_img
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images, plates_positions, bb_img = detect(model1, img)
    labels = classify(model2, images, food_list)
    # print(labels, plates_positions)


TestCamera().run()
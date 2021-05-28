import copy
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont

from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16
from utils.utils import letterbox_image



def get_acc(model_paths):
    get_model_from_name = {
        "mobilenet": MobileNet,
        "resnet50": ResNet50,
        "vgg16": VGG16,
    }
    # model_path_0 = 'logs/resnet_imp_best(53.24).h5'
    model_path_0 = model_paths
    # ----------------------------------------#
    #   预处理训练图片
    # ----------------------------------------#
    def _preprocess_input(x, ):
        x /= 127.5
        x -= 1.
        return x


    # --------------------------------------------#
    #   使用自己训练好的模型预测需要修改4个参数
    #   model_path和classes_path、backbone
    #   和alpha都需要修改！
    # --------------------------------------------#
    class Classification(object):
        _defaults = {
            #"model_path": 'logs/resnet_imp_best(53.24).h5',
            "model_path": 'logs/Resnet_improve/'+model_path_0,
            "classes_path": 'model_data/cls_classes.txt',
            "input_shape": [224, 224, 3],
            "backbone": 'resnet50',
            "alpha": 0.25
        }

        @classmethod
        def get_defaults(cls, n):
            if n in cls._defaults:
                return cls._defaults[n]
            else:
                return "Unrecognized attribute name '" + n + "'"

        # ---------------------------------------------------#
        #   初始化classification
        # ---------------------------------------------------#
        def __init__(self, **kwargs):
            self.__dict__.update(self._defaults)
            self.class_names = self._get_class()
            self.sess = K.get_session()
            self.generate()

        # ---------------------------------------------------#
        #   获得所有的分类
        # ---------------------------------------------------#
        def _get_class(self):
            classes_path = os.path.expanduser(self.classes_path)
            with open(classes_path) as f:
                class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
            return class_names

        # ---------------------------------------------------#
        #   载入模型
        # ---------------------------------------------------#
        def generate(self):
            model_path = os.path.expanduser(self.model_path)
            assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

            # ---------------------------------------------------#
            #   获得种类数量
            # ---------------------------------------------------#
            self.num_classes = len(self.class_names)

            assert self.backbone in ["mobilenet", "resnet50", "vgg16"]

            # ---------------------------------------------------#
            #   载入模型与权值
            # ---------------------------------------------------#

            self.model = get_model_from_name["resnet50"](input_shape=self.input_shape, classes=self.num_classes)
            self.model.load_weights(self.model_path)
            print('{} model, and classes loaded.'.format(model_path))

        # ---------------------------------------------------#
        #   检测图片
        # ---------------------------------------------------#
        def detect_image(self, image):
            old_image = copy.deepcopy(image)
            # ---------------------------------------------------#
            #   对图片进行不失真的resize
            # ---------------------------------------------------#
            crop_img = letterbox_image(image, [self.input_shape[0], self.input_shape[1]])
            photo = np.array(crop_img, dtype=np.float32)

            # ---------------------------------------------------#
            #   图片预处理，归一化
            # ---------------------------------------------------#
            photo = np.reshape(_preprocess_input(photo), [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
            preds = self.model.predict(photo)[0]

            # ---------------------------------------------------#
            #   获得所属种类
            # ---------------------------------------------------#
            class_name = self.class_names[np.argmax(preds)]
            probability = np.max(preds)

            # ---------------------------------------------------#
            #   绘图并写字
            # ---------------------------------------------------#
            plt.subplot(1, 1, 1)
            plt.imshow(np.array(old_image))
            plt.title('Class:%s Probability:%.3f' % (class_name, probability))
            plt.show()
            return class_name

        def close_session(self):
            self.sess.close()

    class top1_Classification(Classification):
        def detect_image(self, image):
            crop_img = letterbox_image(image, [self.input_shape[0],self.input_shape[1]])
            photo = np.array(crop_img,dtype = np.float32)

            # 图片预处理，归一化
            photo = np.reshape(_preprocess_input(photo),[1,self.input_shape[0],self.input_shape[1],self.input_shape[2]])
            preds = self.model.predict(photo)[0]

            arg_pred = np.argmax(preds)
            return arg_pred

    def evaluteTop1(classfication, lines):
        correct = 0
        total = len(lines)
        for index, line in enumerate(lines):
            annotation_path = line.split(';')[1].split('\n')[0]
            x = Image.open(annotation_path)
            y = int(line.split(';')[0])

            pred = classfication.detect_image(x)
            correct += pred == y
            if index % 100 == 0:
                print("[%d/%d]"%(index,total))
        return correct / total

    classfication = top1_Classification()
    with open(r"./cls_test.txt","r") as f:
        lines = f.readlines()
    top1 = evaluteTop1(classfication, lines)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    return top1*100

path = 'D:\PycharmProjects\pythonProject\classification-keras-main\logs\Resnet_improve'
all_acc = []
for i in os.listdir(path):
    acc = get_acc(str(i))
    all_acc.append(acc)
print(all_acc)
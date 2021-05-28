from __future__ import print_function

from keras import backend
import numpy as np
from keras import layers
from keras.layers import (Activation,  BatchNormalization, AveragePooling2D,
                          Conv2D, Dense, Flatten, Input, MaxPooling2D,
                          ZeroPadding2D,GlobalAveragePooling2D)
from keras.models import Model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 减少通道数
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)  # [55,55,64]

    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 3x3卷积
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)  # [55,55,64]

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 上升通道数
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)  # [55,55,256]
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 减少通道数
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)  # [55,55,64]
    y1 = GlobalAveragePooling2D()(x)
    #print('y1:',np.shape(y1)[0])
    #print('x:',np.shape(x))
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 3x3卷积
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)  # [55,55,64]
    y2 = GlobalAveragePooling2D()(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 上升通道数
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)  # [55,55,256]
    y3 = GlobalAveragePooling2D()(x)
    #print(np.shape(y3)[1])
    y = layers.add([y1,y2])
    num = np.shape(y3)[1].value
    y = Dense(num, activation='relu', use_bias=False)(y)
    y = layers.add([y,y3])
    #print(np.shape(y))
    num2 = int(num/8)
    y = Dense(num2, activation='sigmoid', use_bias=False)(y)
    y = Dense(num, activation='relu', use_bias=False)(y)
    #print(np.shape(y))
    #print('y*y3',np.shape(y*y3))
    x = layers.add([x, layers.Multiply()([y,y3])])
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #print(np.shape(x))

    # 残差边
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D((3, 3))(img_input)
    # 224,224,3 -> 112,112,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 112,112,64 -> 56,56,64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 56,56,64 -> 56,56,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 56,56,256 -> 28,28,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 28,28,512 -> 14,14,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 14,14,1024 -> 7,7,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 1,1,2048
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 进行预测
    # 2048
    x = Flatten()(x)

    # num_classes
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='resnet50')

    return model


if __name__ == '__main__':
    model = ResNet50()
    model.summary()

from nets.mobilenet import MobileNet
from nets.resnet50 import ResNet50
from nets.vgg16 import VGG16
from keras.models import load_model
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from utils.utils import letterbox_image
import copy

def _preprocess_input(x,):
    x /= 127.5
    x -= 1.
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def load_image(path):
    img_path = path
    image = Image.open(img_path)
    old_image = copy.deepcopy(image)
    crop_img = letterbox_image(image, [224, 224])
    photo = np.array(crop_img, dtype=np.float32)
    photo = np.reshape(_preprocess_input(photo), [1, 224, 224, 3])

    #x = preprocess_input(x)
    return photo


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def compile_saliency_function(model, activation_layer='res5c_branch2c'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = ResNet50(input_shape=[224,224,3],classes=5)
    return new_model


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # avg
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # create heat map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap



classes = ["kuchu","qinggao", "xianshi", "yinyi","zisi"]
pic_folder = "./img/"
pic_cam_folder = "./img_grad_cam/"
model_path = 'logs/resnet_imp_best(53.24).h5'
#model = ResNet50(input_shape=[224,224,3],classes=5)
#model = load_model('logs/ResNet50_model.h5')
model = ResNet50(input_shape=[224, 224, 3], classes=3)
model.load_weight(model_path)
#model.summary()
list_name = os.listdir(pic_folder)

arr_images = []
for i, file_name in enumerate(list_name):
    top_1 = []
    img = load_image(pic_folder + file_name)
    predictions = model.predict(img)[0]
    print(predictions)
    index = np.argmax(predictions)
    top_1.append(classes[index])
    top_1.append(predictions[index])
    #top_1 = decode_predictions(predictions)[0][0]
    print('Predicted class:')
    print('%s  with probability %.2f' % (top_1[0], top_1[1]))

    predicted_class = np.argmax(predictions)
    cam_image, heat_map = grad_cam(model, img, predicted_class, "res5c_branch2c")

    img_file = image.load_img(pic_folder + list_name[i])
    img_file = image.img_to_array(img_file)

    # guided grad_cam img
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model)

    saliency = saliency_fn([img, 0])
    grad_cam_img = saliency[0] * heat_map[..., np.newaxis]

    # save img
    cam_image = cv2.resize(cam_image, (img_file.shape[1], img_file.shape[0]), cv2.INTER_LINEAR)
    cv2.putText(cam_image,str(top_1[0]), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))
    cv2.putText(cam_image,str(top_1[1]), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))

    grad_cam_img = deprocess_image(grad_cam_img)
    grad_cam_img = cv2.resize(grad_cam_img, (img_file.shape[1], img_file.shape[0]), cv2.INTER_LINEAR)
    cv2.putText(grad_cam_img,str(top_1[0]), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))
    cv2.putText(grad_cam_img,str(top_1[1]), (20, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 255))

    cam_image = cam_image.astype('float32')
    grad_cam_img = grad_cam_img.astype('float32')
    im_h = cv2.hconcat([img_file, cam_image, grad_cam_img])
    cv2.imwrite(pic_cam_folder + list_name[i], im_h)
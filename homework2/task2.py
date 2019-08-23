from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model,Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os
import cv2

def load_vgg16_model(weights_path=None):
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    input_shape = (223, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),    
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(128,(3,3),padding='same',activation='relu'),
        Conv2D(128,(3,3),padding='same',activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(256,(3,3),padding='same',activation='relu'),
        Conv2D(256,(3,3),padding='same',activation='relu'),
        Conv2D(256,(3,3),padding='same',activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512,(3,3),padding='same',activation='relu'),
        Conv2D(512,(3,3),padding='same',activation='relu'),
        Conv2D(512,(3,3),padding='same',activation='relu'),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Conv2D(512,(3,3),padding='same',activation='relu'),
        Conv2D(512,(3,3),padding='same',activation='relu'),
        Conv2D(512,(3,3),padding='same',activation='relu'),        
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax')
    ])
    
    if weights_path:
        model.load_weights(weights_path)
    
    return model
    
    #pass


def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式. 
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    x = x[:,:,::-1] #or x = x.transpose((2,0,1))如果在OpenCV中处理图像，是BGR的顺序。
    #mean += np.sum(x, axis=(0,1).astype(int))
    x -= x.mean()
    return x
    
    #pass


def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    file_names = os.listdir(path)
    for i in file_names[1:]:
        img = cv2.imread(path+"/"+i)
        resize_img = pil_image(target_size).astype(float32)
        return resize_img
    
    #pass


def extract_features(directory, sample_count):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        None
    """
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size = (150, 150),
            batch_size = batch_size,
            class_mode = 'binary')
    i = 0
    for input_batch, labels_batch in generator:
        features_batch = conv_base.predict(input_batch)
        features[i * batch_size : (i+1)*batch_size] = features_batch
        labels[i*batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        
        #注意：这些生成器在循环中不断地生成数据，所以你必须在读取完所有的图像后终止循环
        if i * batch_size >= sample_count:
            break
    return features, labels
    # 参考链接 http://www.zyqit.com/a/pythonpeixun/338.html
    
    #ass


if __name__ == '__main__':
    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = '..\Flicker8k_Dataset'
    features = extract_features(directory)
   # print('提取特征的文件个数：%d' % len(features))
   #print(keras.backend.image_data_format())
    #保存特征到文件
    dump(features, open('features.pkl', 'wb'))




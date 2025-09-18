# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Code Reference: https://github.com/oyam/PanNet-Landsat
Paper References:
    [1] Yang J, Fu X, Hu Y, et al. PanNet: A deep network architecture for pan-sharpening
        [C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 5449-5457.
Notice: I remove the BN layer in the ResBlovck acquiescently, you can use it by removing the uncomment.
"""
import time
import numpy as np
from keras.layers import Concatenate, Conv2D, Input, Layer, Add, Activation, BatchNormalization
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import backend as K
import os
import random
from utils import downgrade_images
import gc
from tensorflow.keras.layers import Layer

# 评价指标
def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels."""
    mse = K.mean(K.square(y_true * 255 - y_pred * 255), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255 / K.sqrt(mse)) / np.log(10))

#高通滤波器可保留信号中的高频成分（快速变化的部分），同时滤除低频成分（平滑或缓慢变化的部分）。在图像处理中，高通滤波器常用于提取边缘、细节等高频信息。
class hp_filter(Layer):
    def __init__(self, **kwargs):
        super(hp_filter, self).__init__(**kwargs)

    def call(self, inputs):
        c = inputs.shape[-1]  # 获取通道数

        # 创建均值滤波核5*5（低通滤波器）它通过对每个像素及其邻域取平均值来平滑图像，从而保留低频成分（平滑区域）。
        kernel = np.ones((5, 5)) / 25.0
        kernel = tf.constant(kernel, dtype=tf.float32)  #

        # 扩展核的维度以匹配输入通道数
        kernel = tf.expand_dims(kernel, -1)  # (5, 5, 1)
        kernel = tf.expand_dims(kernel, -1)  # (5, 5, 1, 1)
        kernel = tf.tile(kernel, (1, 1, c, 1))  # (5, 5, c, 1)

        # 对输入进行深度卷积，得到低频成分
        outs = tf.nn.depthwise_conv2d(inputs, kernel, strides=(1, 1, 1, 1), padding='SAME')

        # 计算高频成分
        outs = inputs - outs

        # 计算输出形状
        self.outs_size = tf.shape(outs)  # 使用 tf.shape 获取动态形状

        return outs

    def compute_output_shape(self, input_shape):
        if not hasattr(self, 'outs_size'):
            # 如果没有计算过输出形状，则返回输入形状
            return input_shape
        return self.outs_size

    def get_config(self):
        config = super(hp_filter, self).get_config()
        return config

class resize1(Layer):
    def __init__(self, target_size, **kwargs):
        self.target_size = (target_size[0], target_size[1])
        super(resize1, self).__init__(**kwargs)

    def call(self, inputs):
        temp = tf.image.resize(inputs, self.target_size, method=tf.image.ResizeMethod.BICUBIC)
        return temp

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[3])

    def get_config(self):
        config = super(resize1, self).get_config()
        return config

def conv_block(inputs, block_name='1'):
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name=block_name + '_1')(inputs)
    #    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name=block_name + '_2')(conv1)
    #    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    outputs = Add()([inputs, conv2])
    return outputs

def pannet(lrms_size=(16, 16, 3), pan_size = (64, 64, 1)):
    #输入LRMS和PAN图像
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    #提取高通信息
    h_lrms = hp_filter()(lrms_inputs)#LRMS高频成分
    h_pan = hp_filter()(pan_inputs)#PAN高频成分
    #将LRMS上采样到PAN尺寸
    re_h_lrms = resize1(pan_size)(h_lrms)#高通后的LRMS上采样
    re_lrms = resize1(pan_size)(lrms_inputs)#原始LRMS上采样
    #融合高频信息
    mixed = Concatenate()([re_h_lrms, h_pan])
    #resnet网络架构（和resnet略有不同）
    mixed1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(mixed)
    x = mixed1
    for i in range(4):
        x = conv_block(x, str(i))
    #最后一层卷积，输出通道数与 LRMS 一致
    x = Conv2D(lrms_size[2], (3, 3), strides=(1, 1), padding='same', name='model1_last1')(x)
    last = Add()([x, re_lrms])
    #构建创建 Keras 模型
    model = Model(inputs = [lrms_inputs, pan_inputs], outputs = last)
    #编译模型，用Adam优化器，损失函数用MAE（我们进行改进的时候可以改为PGL），PSNR作为评价指标
    model.compile(optimizer=Adam(learning_rate = 5e-4), loss = 'mae', metrics=[psnr])
    #打印模型
    model.summary()
    return model

def PanNet(pan, lrms, sensor=None):
    """
    this is an zero-shot learning method with deep learning (PanNet)
    pan: numpy array with MXNXc(M,N,c)
    lrms: numpy array with mxnxC(m,n,C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用 GPU 0
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#不用GPU

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)  # 创建TensorFlow会话
    #K.set_session(sess)  # 将 Keras 后端设置为该会话

    M, N, c = pan.shape
    m, n, C = lrms.shape

    stride = 8  # 训练时裁剪图像的步幅
    training_size = 64  # training patch size
    testing_size = 400  # testing patch size
    reconstructing_size = 320  # reconstructing patch size to avoid boundary effect
    left_pad = (testing_size - reconstructing_size) // 2
    ####计算锐化比率（放大倍数）
    ratio = int(np.round(M / m))  # 放大倍数

    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))  # 确保高度和宽度的比率一致
    ####数据准备和降采样
    train_hrms_all = []
    train_pan_all = []
    train_lrms_all = []

    used_hrms = lrms  # 将初始LRMS作为高分辨率多光谱图像的真值HRMS
    used_lrms = lrms
    # used_lrms：原始LRMS还要降采样之后的图像；used_hrms：不会有尺寸改变
    used_lrms, used_pan = downgrade_images(used_lrms, pan, ratio, sensor=sensor)

    print("LRMS size:",used_lrms.shape)
    print("HRMS size:", used_hrms.shape)
    print("PAN size:", used_pan.shape)
    ####裁剪训练数据
    #used_hrms:HRMS  used_lrms:LRMS  used_pan:PAN
    """crop images"""
    print('croping images...')
    #在训练当中裁剪出来的小块大小为64*64
    for j in range(0, used_pan.shape[0] - training_size, stride):
        for k in range(0, used_pan.shape[1] - training_size, stride):
            temp_hrms = used_hrms[j:j + training_size, k:k + training_size, :]
            temp_pan = used_pan[j:j + training_size, k:k + training_size, :]
            temp_lrms = used_lrms[int(j / 4):int((j + training_size) / 4), int(k / 4):int((k + training_size) / 4), :]

            train_hrms_all.append(temp_hrms)
            train_pan_all.append(temp_pan)
            train_lrms_all.append(temp_lrms)
    ####数据转化为数组并打乱
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_pan_all = np.array(train_pan_all, dtype='float16')
    train_lrms_all = np.array(train_lrms_all, dtype='float16')

    index = [i for i in range(train_hrms_all.shape[0])]
    #    random.seed(2020)
    random.shuffle(index)
    train_hrms = train_hrms_all[index, :, :, :]  # HRMS
    train_pan = train_pan_all[index, :, :, :]  # 降采样后的PAN
    train_lrms = train_lrms_all[index, :, :, :]  # LRMS

    print("train_hrms shape:",train_hrms.shape)
    print("train_pan shape:", train_pan.shape)
    print("train_lrms shape:", train_lrms.shape)

    ####训练网格y
    """train net"""
    print('training...')

    # 定义学习率衰减策略
    def lr_schedule(epoch):
        """Learning Rate Schedule

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    # 保存验证集上PSNR最高的模型权重到PANNET_model.h5
    checkpoint = ModelCheckpoint(filepath='./weights/PANNET_model.h5',
                                 monitor='val_psnr',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]

    model = pannet(lrms_size=(int(training_size / ratio), int(training_size / ratio), C),
                   hrms_size=(training_size, training_size, c))
    # 模型训练50个epoch，批大小32
    model.fit(x=[train_lrms, train_pan],
              y=train_hrms,
              validation_split=0.1,
              batch_size=32,
              epochs=50,
              verbose=1,
              callbacks=callbacks)
    ####测试阶段
    model = pannet(lrms_size=(int(testing_size / ratio), int(testing_size / ratio), C),
                   pan_size=(testing_size, testing_size, c))

    model.load_weights('./weights/PANNET_model.h5')
    ####评估和推理阶段
    """eval"""
    print('evaling...')

    used_lrms = np.expand_dims(lrms, 0)  # (1,m,n,C)
    used_pan = np.expand_dims(pan, 0)  # (1,M,N,c)

    new_M = min(M, m * ratio)
    new_N = min(N, n * ratio)

    print('output image size:', new_M, new_N)

    test_label = np.zeros((new_M, new_N, C), dtype='uint8')  # 初始化输出图像
    # 裁剪和填充输入图像
    used_lrms = used_lrms[:, :new_M // ratio, :new_N // ratio, :]
    used_pan = used_pan[:, :new_M, :new_N, :]

    used_lrms = np.pad(used_lrms, ((0, 0), (left_pad // ratio, testing_size // ratio),
                                   (left_pad // ratio, testing_size // ratio), (0, 0)), mode='symmetric')
    used_pan = np.pad(used_pan, ((0, 0), (left_pad, testing_size),
                                   (left_pad, testing_size), (0, 0)), mode='symmetric')
    # 分块预测并拼接结果
    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrms = used_lrms[:, int(h / ratio):int((h + testing_size) / ratio),
                        int(w / ratio):int((w + testing_size) / ratio), :]
            temp_pan = used_pan[:, h:h + testing_size, w:w + testing_size, :]

            fake = model.predict([temp_lrms, temp_pan])
            fake = np.clip(fake, 0, 1)
            fake.shape = (testing_size, testing_size, C)
            fake = fake[left_pad:(testing_size - left_pad), left_pad:(testing_size - left_pad)]
            fake = np.uint8(fake * 255)

            if h + testing_size > new_M:
                fake = fake[:new_M - h, :, :]

            if w + testing_size > new_N:
                fake = fake[:, :new_N - w, :]

            test_label[h:h + reconstructing_size, w:w + reconstructing_size] = fake

    #    K.clear_session()
    #    gc.collect()
    #    del model

    return np.uint8(test_label)

def PanNettry(pan, lrms, sensor=None):
    """
    this is an zero-shot learning method with deep learning (PanNet)
    hrms: numpy array with MXNXc(M,N,c)
    lrms: numpy array with mxnxC(m,n,C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 指定使用 GPU 0
    M, N, c = pan.shape
    m, n, C = lrms.shape
    stride = 8  # 训练时裁剪图像的步幅
    training_size = 64  # training patch size
    testing_size = 100  # testing patch size
    reconstructing_size = 80  # reconstructing patch size to avoid boundary effect
    left_pad = (testing_size - reconstructing_size) // 2
    ####计算锐化比率（放大倍数）
    ratio = int(np.round(M / m))  # 放大倍数
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))  # 确保高度和宽度的比率一致
    ####数据准备和降采样
    train_hrms_all = []
    train_pan_all = []
    train_lrms_all = []

    used_hrms = lrms  # 将初始LRMS作为高分辨率多光谱图像的真值HRMS
    used_lrms = lrms
    # used_lrhs：原始LRMS还要降采样之后的图像；used_hrms：降采样后的PAN
    used_lrms, used_pan = downgrade_images(used_lrms, pan, ratio, sensor=sensor)

    print("LRMS size:", used_lrms.shape)
    print("HRMS size:", used_hrms.shape)
    print("PAN size:", used_pan.shape)
    ####裁剪训练数据
    # used_hrhs:HRMS  used_lrms:LRMS  used_pan:PAN
    """crop images"""
    print('croping images...')
    # 在训练当中裁剪出来的小块大小为64*64
    for j in range(0, used_pan.shape[0] - training_size, stride):
        for k in range(0, used_pan.shape[1] - training_size, stride):
            temp_hrms = used_hrms[j:j + training_size, k:k + training_size, :]
            temp_pan = used_pan[j:j + training_size, k:k + training_size, :]
            temp_lrms = used_lrms[int(j / 4):int((j + training_size) / 4), int(k / 4):int((k + training_size) / 4), :]

            train_hrms_all.append(temp_hrms)
            train_pan_all.append(temp_pan)
            train_lrms_all.append(temp_lrms)
    ####数据转化为数组并打乱
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_pan_all = np.array(train_pan_all, dtype='float16')
    train_lrms_all = np.array(train_lrms_all, dtype='float16')

    index = [i for i in range(train_hrms_all.shape[0])]
    #    random.seed(2020)
    random.shuffle(index)
    train_hrms = train_hrms_all[index, :, :, :]  # HRMS
    train_pan = train_pan_all[index, :, :, :]  # 降采样后的PAN
    train_lrms = train_lrms_all[index, :, :, :]  # 降采样后的LRMS
    print("Train LRMS size:", train_lrms.shape)#(25,16,16,8)
    print("Train HRMS size:", train_hrms.shape)#(25,64,64,8)
    print("Train PAN size:", train_pan.shape)#(25,64,64,1)

    ####训练网格y
    """train net"""
    print('training...')
    # 定义学习率衰减策略
    def lr_schedule(epoch):
        """Learning Rate Schedule

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    # 保存验证集上PSNR最高的模型权重到PANNET_model.h5
    checkpoint = ModelCheckpoint(filepath='./weights/PANNET_model.h5',
                                 monitor='val_psnr',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    model = pannet(lrms_size=(int(training_size / ratio), int(training_size / ratio), C),
                   pan_size=(training_size, training_size, c))
    # 开始计时
    total_start_time = time.time()
    # 模型训练20个epoch，批大小16
    model.fit(x=[train_lrms, train_pan],
              y=train_hrms,
              validation_split=0.1,
              batch_size=16,
              epochs=20,
              verbose=1,
              callbacks=callbacks)
    total_end_time = time.time()
    total_time = total_end_time - total_start_time  # 总用时（秒）
    total_time_min = total_time / 60  # 转换为分钟
    print(f'Total Runtime: {total_time:.2f} sec ({total_time_min:.2f} min)')
    ####测试阶段
    model = pannet(lrms_size=(int(testing_size / ratio), int(testing_size / ratio), C),
                   pan_size=(testing_size, testing_size, c))

    model.load_weights('./weights/PANNET_model.h5')
    ####评估和推理阶段
    """eval"""
    print('evaling...')
    used_lrms = np.expand_dims(lrms, 0)  # (1,m,n,C)
    used_pan = np.expand_dims(pan, 0)  # (1,M,N,c)
    new_M = min(M, m * ratio)
    new_N = min(N, n * ratio)
    print('output image size:', new_M, new_N)
    test_label = np.zeros((new_M, new_N, C), dtype='uint8')  # 初始化输出图像
    # 裁剪和填充输入图像
    used_lrms = used_lrms[:, :new_M // ratio, :new_N // ratio, :]
    used_pan = used_pan[:, :new_M, :new_N, :]
    used_lrms = np.pad(used_lrms, ((0, 0), (left_pad // ratio, testing_size // ratio),
                                   (left_pad // ratio, testing_size // ratio), (0, 0)), mode='symmetric')
    used_pan = np.pad(used_pan, ((0, 0), (left_pad, testing_size),
                                   (left_pad, testing_size), (0, 0)), mode='symmetric')
    # 分块预测并拼接结果
    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrms = used_lrms[:, int(h / ratio):int((h + testing_size) / ratio),
                        int(w / ratio):int((w + testing_size) / ratio), :]
            temp_pan = used_pan[:, h:h + testing_size, w:w + testing_size, :]

            fake = model.predict([temp_lrms, temp_pan])
            fake = np.clip(fake, 0, 1)
            fake.shape = (testing_size, testing_size, C)
            fake = fake[left_pad:(testing_size - left_pad), left_pad:(testing_size - left_pad)]
            fake = np.uint8(fake * 255)

            if h + testing_size > new_M:
                fake = fake[:new_M - h, :, :]

            if w + testing_size > new_N:
                fake = fake[:, :new_N - w, :]

            test_label[h:h + reconstructing_size, w:w + reconstructing_size] = fake

    #    K.clear_session()
    #    gc.collect()
    #    del model

    return np.uint8(test_label)




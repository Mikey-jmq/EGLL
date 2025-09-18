# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
Code Reference: https://github.com/sergiovitale/pansharpening-cnn-python-version
Paper References:
    Masi G, Cozzolino D, Verdoliva L, et al. Pansharpening by convolutional neural networks
    [J]. Remote Sensing, 2016, 8(7): 594.
"""

import numpy as np
from keras.layers import Concatenate, Conv2D, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
import os
import random
from utils import upsample_interp23, downgrade_images
import gc
import time

def pnn_net(lrms_size=(32, 32, 3), pan_size=(32, 32, 1)):
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)

    mixed = Concatenate()([lrms_inputs, pan_inputs])
    mixed1 = Conv2D(64, (9, 9), strides=(1, 1), padding='same', activation='relu')(mixed)
    mixed1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(mixed1)
    c6 = Conv2D(lrms_size[2], (5, 5), strides=(1, 1), padding='same', activation='relu', name='model1_last1')(mixed1)

    return  Model(inputs=[lrms_inputs, pan_inputs], outputs=c6)

def PNN(pan, lrms, sensor=None):
    """
    this is an zero-shot learning method with deep learning (PNN)
    pan: numpy array with MXNXc
    lrms: numpy array with mxnxC
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    M, N, c = pan.shape
    m, n, C = lrms.shape
    stride = 8
    training_size = 64  # training patch size
    testing_size = 100  # testing patch size
    reconstructing_size = 80  # reconstructing patch size
    left_pad = (testing_size - reconstructing_size) // 2
    ratio = int(np.round(M / m))
    assert int(np.round(M / m)) == int(np.round(N / n))
    train_hrms_all = []
    train_pan_all = []
    train_lrms_all = []
    used_hrms = lrms
    used_lrms = lrms
    used_lrms, used_pan = downgrade_images(used_lrms, pan, ratio, sensor=sensor)
    used_lrms = upsample_interp23(used_lrms, ratio)
    """crop images"""
    print('croping images...')
    for j in range(0, used_pan.shape[0] - training_size, stride):
        for k in range(0, used_pan.shape[1] - training_size, stride):
            temp_hrms = used_hrms[j:j + training_size, k:k + training_size, :]
            temp_pan = used_pan[j:j + training_size, k:k + training_size, :]
            temp_lrms = used_lrms[j:j + training_size, k:k + training_size, :]
            train_hrms_all.append(temp_hrms)
            train_pan_all.append(temp_pan)
            train_lrms_all.append(temp_lrms)
    train_hrms_all = np.array(train_hrms_all, dtype='float16')
    train_pan_all = np.array(train_pan_all, dtype='float16')
    train_lrms_all = np.array(train_lrms_all, dtype='float16')
    index = [i for i in range(train_hrms_all.shape[0])]
    #    random.seed(2020)
    random.shuffle(index)
    train_hrms = train_hrms_all[index, :, :, :]
    train_pan = train_pan_all[index, :, :, :]
    train_lrms = train_lrms_all[index, :, :, :]
    """train net"""
    print('training...')
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
    checkpoint = ModelCheckpoint(filepath='./weights/PNN_model.h5',
                                 monitor='val_psnr',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    model = pnn_net(lrms_size=(training_size, training_size, C), pan_size=(training_size, training_size, c))
    # 开始计时
    total_start_time = time.time()
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

    #测试阶段
    model = pnn_net(lrms_size=(testing_size, testing_size, C), pan_size=(testing_size, testing_size, c))
    model.load_weights('./weights/PNN_model.h5')
    """eval"""
    print('evaling...')
    new_M = min(M, m * ratio)
    new_N = min(N, n * ratio)
    test_label = np.zeros((new_M, new_N, C), dtype='uint8')
    used_lrms = lrms[:new_M // ratio, :new_N // ratio, :]
    used_pan = pan[:new_M, :new_N, :]
    used_lrms = upsample_interp23(used_lrms, ratio)
    used_lrms = np.expand_dims(used_lrms, 0)
    used_pan = np.expand_dims(used_pan, 0)
    used_lrms = np.pad(used_lrms, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
                       mode='symmetric')
    used_pan = np.pad(used_pan, ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
                       mode='symmetric')
    for h in tqdm(range(0, new_M, reconstructing_size)):
        for w in range(0, new_N, reconstructing_size):
            temp_lrms = used_lrms[:, h:h + testing_size, w:w + testing_size, :]
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
    # K.clear_session()
    # gc.collect()
    # del model
    return np.uint8(test_label)
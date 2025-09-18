import numpy as np
from keras.layers import Concatenate, Conv2D, Input, Layer, Add, Activation, BatchNormalization,Lambda,ReLU,Dense,GlobalAveragePooling2D
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

class LAGConv2D(Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding='same', dilation=1, groups=1, use_bias=True, **kwargs):
        super(LAGConv2D, self).__init__(**kwargs)
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

        # attention1: 局部自适应权重
        self.attention1 = [
            Conv2D(kernel_size ** 2, kernel_size, strides=stride, padding=padding),
            ReLU(),
            Conv2D(kernel_size ** 2, 1, strides=1, padding='valid'),
            ReLU(),
            Conv2D(kernel_size ** 2, 1, strides=1, padding='valid'),
            Activation('sigmoid')
        ]

        # attention3: 全局偏置
        if use_bias:
            self.attention3 = [
                GlobalAveragePooling2D(),
                # 使用 Lambda 层将 (B, in_planes) 重塑为 (B, 1, 1, in_planes)
                Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)),
                Conv2D(out_planes, 1, strides=1, padding='valid'),
                ReLU(),
                Conv2D(out_planes, 1, strides=1, padding='valid')
            ]

        # 标准卷积层
        self.conv = Conv2D(out_planes, kernel_size, strides=stride, padding=padding,
                          dilation_rate=dilation, groups=groups, use_bias=False)

    def build(self, input_shape):
        # 构建 conv 层
        self.conv.build(input_shape)
        self.weight = self.conv.kernel

        # 构建 attention1 层
        x = input_shape
        for layer in self.attention1:
            layer.build(x)
            x = layer.compute_output_shape(x)

        # 构建 attention3 层
        if self.use_bias:
            x = input_shape
            for layer in self.attention3:
                layer.build(x)
                x = layer.compute_output_shape(x)

        super(LAGConv2D, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        H, W = inputs.shape[1], inputs.shape[2]
        k = self.kernel_size
        n = self.in_planes
        m = self.out_planes

        # 计算输出尺寸
        n_H = 1 + int((H + 2 * (k // 2) - k) // self.stride)
        n_W = 1 + int((W + 2 * (k // 2) - k) // self.stride)

        # attention1：生成局部权重
        x = inputs
        for layer in self.attention1:
            x = layer(x)  # (B, n_H, n_W, k*k)
        atw1 = x

        # 调整 atw1 形状
        atw1 = tf.transpose(atw1, [0, 1, 2, 3])  # (B, n_H, n_W, k*k)
        atw1 = tf.expand_dims(atw1, axis=3)  # (B, n_H, n_W, 1, k*k)
        atw1 = tf.tile(atw1, [1, 1, 1, n, 1])  # (B, n_H, n_W, n, k*k)
        atw1 = tf.reshape(atw1, [batch_size, n_H, n_W, n * k * k])  # (B, n_H, n_W, n*k*k)
        atw1 = tf.reshape(atw1, [batch_size, n_H * n_W, n * k * k])  # (B, n_H*n_W, n*k*k)
        atw1 = tf.transpose(atw1, [0, 2, 1])  # (B, n*k*k, n_H*n_W)

        # 提取滑动窗口特征
        kx = tf.image.extract_patches(
            images=inputs,
            sizes=[1, k, k, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, self.dilation, self.dilation, 1],
            padding=self.padding.upper()
        )  # (B, n_H, n_W, n*k*k)
        kx = tf.reshape(kx, [batch_size, n_H * n_W, n * k * k])  # (B, n_H*n_W, n*k*k)
        kx = tf.transpose(kx, [0, 2, 1])  # (B, n*k*k, n_H*n_W)

        # 自适应卷积
        atx = atw1 * kx  # (B, n*k*k, n_H*n_W)
        atx = tf.transpose(atx, [0, 2, 1])  # (B, n_H*n_W, n*k*k)
        atx = tf.reshape(atx, [1, batch_size * n_H * n_W, n * k * k])  # (1, B*n_H*n_W, n*k*k)

        # 卷积权重处理
        w = tf.reshape(self.weight, [k * k * n, m])  # (n*k*k, m)
        y = tf.matmul(atx, w)  # (1, B*n_H*n_W, m)
        y = tf.reshape(y, [batch_size, n_H * n_W, m])  # (B, n_H*n_W, m)

        # 添加全局偏置
        if self.use_bias:
            bias = inputs
            for layer in self.attention3:
                bias = layer(bias)  # (B, 1, 1, m)
            bias = tf.reshape(bias, [batch_size, m, 1])  # (B, m, 1)
            bias = tf.tile(bias, [1, 1, n_H * n_W])  # (B, m, n_H*n_W)
            bias = tf.transpose(bias, [0, 2, 1])  # (B, n_H*n_W, m)
            y = y + bias

        # 重建输出（Keras 标准形状）
        y = tf.reshape(y, [batch_size, n_H, n_W, m])  # (B, n_H, n_W, m)
        return y

    def compute_output_shape(self, input_shape):
        H, W = input_shape[1], input_shape[2]
        n_H = 1 + int((H + 2 * (self.kernel_size // 2) - self.kernel_size) // self.stride)
        n_W = 1 + int((W + 2 * (self.kernel_size // 2) - self.kernel_size) // self.stride)
        return (input_shape[0], n_H, n_W, self.out_planes)

# LACRB 残差块
def LACRB(inputs, in_planes):
    x = LAGConv2D(in_planes, in_planes, kernel_size=3, stride=1, padding='same', use_bias=True)(inputs)
    x = ReLU()(x)
    x = LAGConv2D(in_planes, in_planes, kernel_size=3, stride=1, padding='same', use_bias=True)(x)
    outputs = Add()([inputs, x])
    return outputs

# LAGNET 模型
def lagnet(lrms_size=(16, 16, 8), pan_size=(64, 64, 1)):
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    re_lrms = resize1(pan_size)(lrms_inputs)  # (H, W, C)

    x = Concatenate()([re_lrms, pan_inputs])# (H, W, 1+C)
    # Head 卷积
    x = LAGConv2D(1 + lrms_size[-1], 32, kernel_size=3, stride=1, padding='same', use_bias=True)(x)
    x = ReLU()(x)
    # 5 个残差块
    for i in range(5):
        x = LACRB(x, in_planes=32)
    # Tail 卷积
    x = LAGConv2D(32, lrms_size[-1], kernel_size=3, stride=1, padding='same', use_bias=True)(x)

    outputs = Add()([x, re_lrms])
    return Model(inputs=[lrms_inputs, pan_inputs], outputs=outputs)

def LAGConv(pan, lrms, sensor=None):
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
    ####裁剪训练数据
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
    random.shuffle(index)
    train_hrms = train_hrms_all[index, :, :, :]  # HRMS(25,64,64,8)
    train_pan = train_pan_all[index, :, :, :]  # 降采样后的PAN(25,64,64,1)
    train_lrms = train_lrms_all[index, :, :, :]  # 降采样后的LRMS(25,16,16,8)

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
    checkpoint = ModelCheckpoint(filepath='./weights/LAGConv_model.h5',
                                 monitor='val_psnr',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    model = lagnet(lrms_size=(int(training_size / ratio), int(training_size / ratio), C),
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
    model = lagnet(lrms_size=(int(testing_size / ratio), int(testing_size / ratio), C),
                   pan_size=(testing_size, testing_size, c))

    model.load_weights('./weights/LAGConv_model.h5')
    ####评估和推理阶段
    """eval"""
    print('evaling...')
    used_lrms = np.expand_dims(lrms, 0)  # (1,m,n,C)
    used_pan = np.expand_dims(pan, 0)  # (1,M,N,c)
    new_M = min(M, m * ratio)
    new_N = min(N, n * ratio)
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

# if __name__ == "__main__":
#     training_size = 64
#     ratio = 4
#     C = 8
#     c = 1
#     model = lagnet(lrms_size=(int(training_size / ratio), int(training_size / ratio), C),
#                      pan_size=(training_size, training_size, c))
#     lrms = np.random.randn(1, 16, 16, 8).astype(np.float32)
#     pan = np.random.randn(1, 64, 64, 1).astype(np.float32)
#     output = model.predict([lrms, pan])
#     print("Output shape:", output.shape)  # 应为 (1, 64, 64, 8)

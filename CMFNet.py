import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Input, Dropout, Concatenate, Add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import random
from utils import upsample_interp23, downgrade_images
import gc
import time

class LayerNorm2d(Layer):
    def __init__(self, channels, eps=1e-6, **kwargs):
        super(LayerNorm2d, self).__init__(**kwargs)
        self.channels = channels
        self.eps = eps
        self.weight = self.add_weight(name='weight', shape=(channels,), initializer='ones', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(channels,), initializer='zeros', trainable=True)

    def call(self, x):
        mu = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.reduce_mean(tf.square(x - mu), axis=-1, keepdims=True)
        y = (x - mu) / tf.sqrt(var + self.eps)
        weight_reshaped = tf.reshape(self.weight, (1, 1, 1, self.channels))
        bias_reshaped = tf.reshape(self.bias, (1, 1, 1, self.channels))

        return y * weight_reshaped + bias_reshaped

    def compute_output_shape(self, input_shape):
        return input_shape

class SimpleGate(Layer):
    def call(self, x):
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        return x1 * x2

class NAFBlock(Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, **kwargs):
        super(NAFBlock, self).__init__(**kwargs)
        dw_channel = c * DW_Expand#512
        self.conv1 = Conv2D(dw_channel, 1, padding='same', use_bias=True)
        self.conv2 = Conv2D(dw_channel, 3, padding='same', groups=dw_channel, use_bias=True)
        self.conv3 = Conv2D(c, 1, padding='same', use_bias=True)  # 输出 c 通道

        self.sca_avg = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True),
            Conv2D(dw_channel // 4, 1, padding='same', use_bias=True)
        ])
        self.sca_max = tf.keras.Sequential([
            tf.keras.layers.GlobalMaxPooling2D(keepdims=True),
            Conv2D(dw_channel // 4, 1, padding='same', use_bias=True)
        ])

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = Conv2D(ffn_channel, 1, padding='same', use_bias=True)
        self.conv5 = Conv2D(c, 1, padding='same', use_bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = Dropout(drop_out_rate) if drop_out_rate > 0.0 else tf.keras.layers.Lambda(lambda x: x)
        self.dropout2 = Dropout(drop_out_rate) if drop_out_rate > 0.0 else tf.keras.layers.Lambda(lambda x: x)

        self.beta = self.add_weight(name='beta', shape=(1, 1, 1,c), initializer='zeros', trainable=True)
        self.gamma = self.add_weight(name='gamma', shape=(1, 1, 1,c), initializer='zeros', trainable=True)

    def call(self, x):
        y = x#(None,16,16,128)
        x = self.norm1(x)  # (None,16,16,256)
        x = self.conv1(x)#(None,16,16,256)
        x = self.conv2(x)#(None,16,16,256)
        x = self.sg(x)#(None,16,16,128)
        x_avg, x_max = tf.split(x, num_or_size_splits=2, axis=-1)#(None,16,16,64),(None,16,16,64)
        x_avg = self.sca_avg(x_avg) * x_avg#(None,16,16,64)
        x_max = self.sca_max(x_max) * x_max#(None,16,16,64)
        x = Concatenate(axis=-1)([x_avg, x_max])#(None,16,16,128)
        x = self.conv3(x)#(None,16,16,128)
        x = self.dropout1(x)#(None,16,16,128)
        y = y + x * self.beta#(None,16,16,128)

        x = self.norm2(y)#(None,16,16,128)
        x = self.conv4(x)#(None,16,16,128)
        x = self.sg(x)#(None,16,16,128)
        x = self.conv5(x)#(None,16,16,128)
        x = self.dropout2(x)#(None,16,16,128)
        return y + x * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape

class MultiscalePANEncoder(Layer):
    def __init__(self, in_channel, width=32, enc_blk_nums=[1, 1], middle_blk_num=1, **kwargs):
        super(MultiscalePANEncoder, self).__init__(**kwargs)
        self.intro = Conv2D(width, 3, padding='same', strides=1)
        self.encs = []
        self.downs = []
        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encs.append(tf.keras.Sequential([NAFBlock(chan) for _ in range(num)]))
            self.downs.append(Conv2D(chan * 2, 2, strides=2))
            chan = chan * 2
        self.middle_blks = tf.keras.Sequential([NAFBlock(chan) for _ in range(middle_blk_num)])

    def call(self, pan):
        pan = self.intro(pan)#(none,64,64,32)
        encs = []
        for enc, down in zip(self.encs, self.downs):
            pan = enc(pan)
            encs.append(pan)
            pan = down(pan)
        pan = self.middle_blks(pan)
        encs.append(pan)
        return encs

class MultiscaleMSEncoder(Layer):
    def __init__(self, in_channel, width=32, enc_blk_nums=[1, 1], middle_blk_num=1, **kwargs):
        super(MultiscaleMSEncoder, self).__init__(**kwargs)
        self.intro = Conv2D(width * 4, 3, padding='same', strides=1)  # 适配 in_channel=8
        self.encs = []
        self.ups = []

        chan = width * 4
        for num in enc_blk_nums:
            self.encs.append(tf.keras.Sequential([NAFBlock(chan) for _ in range(num)]))
            self.ups.append(tf.keras.Sequential([
                Conv2D(chan * 2, 1, use_bias=False),
                tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))  # 等同于PixelShuffle
            ]))
            chan = chan // 2

        self.middle_blks = tf.keras.Sequential([NAFBlock(chan) for _ in range(middle_blk_num)])

    def call(self, ms):
        ms = self.intro(ms)#(None,16,16,128)
        encs = []
        for enc, up in zip(self.encs, self.ups):
            ms = enc(ms)
            encs.append(ms)
            ms = up(ms)
        ms = self.middle_blks(ms)
        encs.append(ms)
        return encs[::-1]

class UEDM(Model):
    def __init__(self, img_channel=8, width=32, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1], **kwargs):
        super(UEDM, self).__init__(**kwargs)
        self.ending = Conv2D(img_channel, 3, padding='same', strides=1, use_bias=True)
        self.encoders = []
        self.downs = []
        self.ups = []
        self.decoders = []
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(tf.keras.Sequential([NAFBlock(chan) for _ in range(num)]))
            self.downs.append(Conv2D(chan * 2, 2, strides=2))
            chan = chan * 2

        self.middle_blks = tf.keras.Sequential([NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(tf.keras.Sequential([
                Conv2D(chan * 2, 1, use_bias=False),
                tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))
            ]))
            chan = chan // 2
            self.decoders.append(tf.keras.Sequential([NAFBlock(chan) for _ in range(num)]))

        self.ms_encs = MultiscaleMSEncoder(img_channel, width, enc_blk_nums, middle_blk_num)
        self.pan_encs = MultiscalePANEncoder(1, width, enc_blk_nums, middle_blk_num)

    def call(self, inputs):
        ms, pan = inputs
        ms_encs = self.ms_encs(ms)
        pan_encs = self.pan_encs(pan)
        fuse = 0
        for encoder, down, ms, pan in zip(self.encoders, self.downs, ms_encs[:2], pan_encs[:2]):
            fuse = ms + pan + fuse
            fuse = encoder(fuse)
            fuse = down(fuse)
        fuse = fuse + ms_encs[-1] + pan_encs[-1]
        fuse = self.middle_blks(fuse)

        for decoder, up, ms, pan in zip(self.decoders, self.ups, ms_encs[::-1][1:], pan_encs[::-1][1:]):
            fuse = up(fuse)
            fuse = fuse + ms + pan
            fuse = decoder(fuse)

        fuse = self.ending(fuse)
        return fuse

def cmfnet(lrms_size=(16, 16, 8), pan_size=(64, 64, 1)):
    ms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    model = UEDM(img_channel=lrms_size[-1])
    outputs = model([ms_inputs, pan_inputs])
    return Model(inputs=[ms_inputs, pan_inputs], outputs=outputs)

def CMFNet(pan, lrms, sensor=None):
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
    checkpoint = ModelCheckpoint(filepath='./weights/CMFNet_model.h5',
                                 monitor='val_psnr',
                                 mode='max',
                                 verbose=1,
                                 save_best_only=True)
    callbacks = [lr_scheduler, checkpoint]
    model = cmfnet(lrms_size=(int(training_size / ratio), int(training_size / ratio), C),
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
    model = cmfnet(lrms_size=(int(testing_size / ratio), int(testing_size / ratio), C),
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
# 示例调用
if __name__ == "__main__":
    model = cmfnet()
    ms = tf.random.normal([1,16, 16, 8])
    pan = tf.random.normal([1,64, 64, 1])
    out = model.predict([ms, pan])
    print(out.shape)
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras import backend as K
import random
import math
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add , Lambda, Activation,ReLU
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os
from PanNet import hp_filter, resize1, conv_block
from GPPNN import basic_unit,lr_block,pan_block
import time
from utils import upsample_interp23,downgrade_images
from MSDCNN import MSRResB,ShallowBranch
from LAGConv import LAGConv2D,LACRB
from CMFNet import UEDM
from evaluation import *
from tensorflow.keras.callbacks import Callback

#MS-SSIM+L1损失
def gaussian_kernel(size=5, sigma=1.0, channels=3):
    """生成2D高斯核"""
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / tf.reduce_sum(g)  # 归一化
    kernel = tf.tensordot(g, g, axes=0)  # 外积生成2D核
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    return tf.tile(kernel, [1, 1, channels, 1])

class MSSSIML1(tf.keras.losses.Loss):
    def __init__(self,
                 alpha=0.025,
                 sigmas=[0.5, 1., 2., 4., 8.],
                 C1=0.01 ** 2,
                 C2=0.03 ** 2,
                 kernel_size=5,
                 **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.sigmas = sigmas
        self.C1 = C1
        self.C2 = C2
        self.kernel_size = kernel_size

    def _gaussian_blur(self, x, sigma):
        """应用高斯模糊"""
        kernel = gaussian_kernel(size=self.kernel_size, sigma=sigma, channels=x.shape[-1])
        return tf.nn.depthwise_conv2d(
            x, kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )

    def _compute_ssim(self, x, y, sigma):
        """单尺度SSIM计算"""
        # 高斯滤波
        ux = self._gaussian_blur(x, sigma)
        uy = self._gaussian_blur(y, sigma)

        # 方差与协方差
        uxx = self._gaussian_blur(x * x, sigma)
        uyy = self._gaussian_blur(y * y, sigma)
        uxy = self._gaussian_blur(x * y, sigma)

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        # SSIM计算
        luminance = (2 * ux * uy + self.C1) / (ux ** 2 + uy ** 2 + self.C1)
        contrast = (2 * vxy + self.C2) / (vx + vy + self.C2)

        return luminance * contrast

    def call(self, y_true, y_pred):
        # 输入预处理
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 计算多尺度SSIM
        ms_ssim = []
        for sigma in self.sigmas:
            ssim_map = self._compute_ssim(y_pred, y_true, sigma)
            ms_ssim.append(tf.reduce_mean(ssim_map, axis=[1, 2, 3]))

        # 多尺度乘积组合
        ms_ssim = tf.stack(ms_ssim, axis=1)
        ms_ssim = tf.reduce_prod(ms_ssim, axis=1)
        ms_ssim_loss = 1.0 - tf.reduce_mean(ms_ssim)

        # 高斯加权L1计算（使用最大sigma对应的核）
        kernel = gaussian_kernel(size=self.kernel_size,
                                 sigma=self.sigmas[-1],
                                 channels=y_true.shape[-1])
        abs_diff = tf.abs(y_pred - y_true)
        l1_loss = tf.reduce_mean(
            tf.nn.depthwise_conv2d(
                abs_diff, kernel,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
        )

        # 组合损失
        return self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss

#Adaptive Robust Loss
class AdaptiveLossFunction(Layer):
    def __init__(self, num_dims, alpha_lo=0.001, alpha_hi=1.999, scale_lo=1e-5, scale_init=1.0, **kwargs):
        super(AdaptiveLossFunction, self).__init__(**kwargs)
        self.num_dims = num_dims
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.scale_lo = scale_lo
        self.scale_init = scale_init

    def build(self, input_shape):
        # Alpha 参数初始化
        if self.alpha_lo == self.alpha_hi:
            self.alpha = self.add_weight(
                name='alpha', shape=(1, self.num_dims),
                initializer=tf.constant_initializer(self.alpha_lo), trainable=False)
        else:
            alpha_init = (self.alpha_lo + self.alpha_hi) / 2.0
            latent_alpha_init = self._inv_affine_sigmoid(alpha_init)
            self.latent_alpha = self.add_weight(
                name='latent_alpha', shape=(1, self.num_dims),
                initializer=tf.constant_initializer(latent_alpha_init), trainable=True)

        # Scale 参数初始化
        if self.scale_lo == self.scale_init:
            self.scale = self.add_weight(
                name='scale', shape=(1, self.num_dims),
                initializer=tf.constant_initializer(self.scale_init), trainable=False)
        else:
            self.latent_scale = self.add_weight(
                name='latent_scale', shape=(1, self.num_dims),
                initializer=tf.zeros_initializer(), trainable=True)

        super(AdaptiveLossFunction, self).build(input_shape)

    def __call__(self, y_true, y_pred):
        if not self.built:
            input_shape = tf.shape(y_pred)
            self.build(input_shape)

        x = y_pred - y_true
        x = tf.reshape(x, [-1, self.num_dims])

        alpha = self._get_alpha()
        scale = self._get_scale()

        abs_x = tf.abs(x)
        loss = (tf.pow(abs_x, alpha) / alpha + tf.math.log(scale)) / scale
        return tf.reduce_mean(loss)

    def _get_alpha(self):
        if self.alpha_lo == self.alpha_hi:
            return self.alpha
        else:
            return self._affine_sigmoid(self.latent_alpha)

    def _get_scale(self):
        if self.scale_lo == self.scale_init:
            return self.scale
        else:
            return self._affine_softplus(self.latent_scale)

    def _affine_sigmoid(self, logits):
        return tf.sigmoid(logits) * (self.alpha_hi - self.alpha_lo) + self.alpha_lo

    def _inv_affine_sigmoid(self, alpha):
        return math.log(alpha - self.alpha_lo) - math.log(self.alpha_hi - alpha)

    def _affine_softplus(self, x):
        shift = tf.math.log(tf.exp(1.0) - 1.0)
        return (self.scale_init - self.scale_lo) * tf.nn.softplus(x + shift) + self.scale_lo

#EGLL
def modified_log_sobel(alpha=6, kappa=2,beta1=0.9,beta2=0.5, epsilon=1e-6):
    """
        alpha: 对数函数的整体尺度
        kappa:对数函数的增长速度
        beta1:梯度一致项的权重
        beta2:mae的权重
        """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        pixel_diff = y_pred - y_true
        abs_error = tf.abs(pixel_diff)
        # 修正对数损失：log(1 + alpha * |x|^kappa)
        log_term = tf.reduce_mean(alpha * tf.pow(abs_error + epsilon, kappa))
        # 梯度一致性项：
        grad_term = tf.reduce_mean(tf.abs(tf.image.sobel_edges(y_pred)-tf.image.sobel_edges(y_true)))
        # L1 损失项
        l1_loss = tf.reduce_mean(abs_error)
        total_loss = tf.reduce_mean(tf.math.log(1+log_term+beta1*grad_term))+ beta2 * l1_loss

        return total_loss

    return loss

def set_global_seed(seed=30):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def split_dataset(pan, lrms, size=256, train_ratio=0.8, val_ratio=0.1, stride=32,seed=30):
    """
    将PAN和MS图像按指定大小和步幅切割，打乱后分配给训练、验证、测试集
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    size: 切割块大小（PAN图像的尺寸，默认为64）
    train_ratio: 训练集比例（默认0.8）
    val_ratio: 验证集比例（默认0.1）
    stride: 切割步幅（默认8）
    返回: 训练、验证、测试集的PAN和LRMS网格
    """
    set_global_seed(seed)  # 设置种子
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))  # 放大倍数
    assert int(np.round(M / m)) == int(np.round(N / n))
    pan_grids, lrms_grids = [], []
    # 按步幅切割PAN和LRMS图像（不降采样）
    for j in range(0, M - size, stride):
        for k in range(0, N - size, stride):
            temp_pan = pan[j:j + size, k:k + size, :]
            temp_lrms = lrms[int(j / ratio):int((j + size) / ratio), int(k / ratio):int((k + size) / ratio), :]
            pan_grids.append(temp_pan)
            lrms_grids.append(temp_lrms)

    # 转换为numpy数组
    pan_grids = np.array(pan_grids, dtype='float16')
    lrms_grids = np.array(lrms_grids, dtype='float16')

    # 打乱网格
    total_grids = len(pan_grids)
    indices = list(range(total_grids))
    random.shuffle(indices)

    # 按比例分配
    num_train = int(total_grids * train_ratio)
    num_val = int(total_grids * val_ratio)
    num_test = total_grids - num_train - num_val

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    # 提取训练、验证、测试集网格
    train_pan_grids = pan_grids[train_indices]
    train_lrms_grids = lrms_grids[train_indices]
    val_pan_grids = pan_grids[val_indices]
    val_lrms_grids = lrms_grids[val_indices]
    test_pan_grids = pan_grids[test_indices]
    test_lrms_grids = lrms_grids[test_indices]
    print(f"Shuffled indices: {indices[:10]}")  # 打印前10个索引

    return (train_pan_grids, train_lrms_grids), \
       (val_pan_grids, val_lrms_grids), \
       (test_pan_grids, test_lrms_grids)

"""PanNet"""
def pannet(lrms_size=(16, 16, 3), pan_size=(64, 64, 1)):
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)

    h_lrms = hp_filter()(lrms_inputs)
    h_pan = hp_filter()(pan_inputs)

    re_h_lrms = resize1(pan_size)(h_lrms)
    re_lrms = resize1(pan_size)(lrms_inputs)

    mixed = Concatenate()([re_h_lrms, h_pan])
    mixed1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(mixed)

    x = mixed1
    for i in range(4):
        x = conv_block(x, str(i))

    x = Conv2D(lrms_size[2], (3, 3), strides=(1, 1), padding='same')(x)
    last = Add()([x, re_lrms])

    return Model(inputs=[lrms_inputs, pan_inputs], outputs=last)

def PanNet(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'log': modified_log_sobel(),
        # 'ms-ssim+l1': MSSSIML1(alpha=0.025,sigmas=[0.5, 1.0, 2.0], kernel_size=5),
        # "Ada loss":AdaptiveLossFunction(num_dims=C*64*64),
        'mse': 'mse'
    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=25)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor='WV2')
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据（仅一次）
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio)
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'PANNET_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = pannet(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = pannet(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './PanNet_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nPanNet Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nPanNet Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""PNN"""
def pnn_net(lrms_size=(32, 32, 3), pan_size=(32, 32, 1)):
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)

    mixed = Concatenate()([lrms_inputs, pan_inputs])
    mixed1 = Conv2D(64, (9, 9), strides=(1, 1), padding='same', activation='relu')(mixed)
    mixed1 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu')(mixed1)
    c6 = Conv2D(lrms_size[2], (5, 5), strides=(1, 1), padding='same', activation='relu', name='model1_last1')(mixed1)

    return  Model(inputs=[lrms_inputs, pan_inputs], outputs=c6)

def PNN(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'log': modified_log_sobel(),
        'mse': 'mse'

    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=20)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor='WV2')
            lrms_upsample = upsample_interp23(lrms_down,ratio)
            used_pan.append(pan_down)
            used_lrms.append(lrms_upsample)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据（仅一次）
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio)
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'PNN_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = pnn_net(lrms_size=(64, 64, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        test_lrms_upsampled = np.array([upsample_interp23(lrms, ratio) for lrms in test_lrms], dtype='float16')
        model = pnn_net(lrms_size=(256, 256, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms_upsampled, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './PNN_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nPNN Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nPNN Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""FusionNet"""
def fusionnet(lrms_size=(16, 16, 3), pan_size=(32, 32, 1)):
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    re_lrms = resize1(pan_size)(lrms_inputs)
    pan_concat = Lambda(lambda x: tf.repeat(x, lrms_size[-1], axis=-1))(pan_inputs)#扩展通道(用的是Keras)
    mixed = Lambda(lambda x: x[0] - x[1])([pan_concat, re_lrms])
    mixed1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(mixed)
    x = mixed1
    for i in range(4):
        x = conv_block(x, str(i))
    x = Conv2D(lrms_size[2], (3, 3), strides=(1, 1), padding='same')(x)
    last = Add()([x, re_lrms])

    return Model(inputs=[lrms_inputs, pan_inputs], outputs=last)

def FusionNet(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'log': modified_log_sobel(),
        'mse': 'mse'

    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=29)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor='WV2')
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据（仅一次）
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio)
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'FusionNet_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = fusionnet(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = fusionnet(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './FusionNet_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nFusionNet Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nFusionNet Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""MSDCNN"""
def msdcnn(lrms_size=(16, 16, 3), pan_size=(32, 32, 1)):
    lrms_inputs = Input(lrms_size)  # (16,16,3)
    pan_inputs = Input(pan_size)  # (32,32,1)
    re_lrms = resize1(pan_size)(lrms_inputs) #(32,32,3)
    mixed = Concatenate()([re_lrms, pan_inputs]) #(32,32,4)
    #用MSResB
    x = Conv2D(60, (7, 7), padding='same', activation='relu')(mixed)#(32,32,64)
    x = MSRResB(x, filters=20, name='msrresb1')#
    x = Conv2D(30, (3, 3), padding='same', activation='relu')(x)
    x = MSRResB(x, filters=10, name='msrresb2')
    x = Conv2D(lrms_size[-1], (5, 5), padding='same')(x)
    #浅层特征分支
    shallow = ShallowBranch(mixed, output_channels=lrms_size[-1])
    #特征融合
    last = Activation('relu')(Add()([x, shallow]))
    return Model(inputs=[lrms_inputs, pan_inputs], outputs=last)

def MSDCNN(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'mse': 'mse',
        'log': modified_log_sobel()
    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=26)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio, sensor=None):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor=sensor)
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio, sensor='WV2')
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio, sensor='WV2')
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'MSDCNN_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = msdcnn(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = msdcnn(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './MSDCNN_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nMSDCNN Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nMSDCNN Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""GPPNN"""
def gppnn(lrms_size=(16, 16, 3), pan_size = (64, 64, 1)):
    # 输入LRMS和PAN图像
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    hr = resize1(pan_size)(lrms_inputs)
    for i in range(4):
        hr = lr_block(hr, lrms_inputs,64, name=f'lrblock_{i + 1}')
        hr = pan_block(hr, pan_inputs, 64,  name=f'panblock_{i + 1}')
    return Model(inputs=[lrms_inputs, pan_inputs], outputs=hr)

def GPPNN(pan, lrms,epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'mse': 'mse',
        'log': modified_log_sobel()
    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=23)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio, sensor='WV2'):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor=sensor)
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio, sensor='WV2')
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio, sensor='WV2')
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'GPPNN_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = gppnn(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = gppnn(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './GPPNN_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nGPPNN Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nGPPNN Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""LAGConv"""
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

def LAGConv(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'log': modified_log_sobel(),
        'mse': 'mse'

    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=27)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor='WV2')
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据（仅一次）
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio)
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'LAGConv_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = lagnet(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = lagnet(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './LAGConv_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nLAGConv Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nLAGConv Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

"""CMFNet"""
def cmfnet(lrms_size=(16, 16, 8), pan_size=(64, 64, 1)):
    ms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    model = UEDM(img_channel=lrms_size[-1])
    outputs = model([ms_inputs, pan_inputs])
    return Model(inputs=[ms_inputs, pan_inputs], outputs=outputs)

def CMFNet(pan, lrms, epoch=100):
    """
    统一的PanNet函数，遍历多个损失函数进行训练、验证、测试
    pan: numpy array with (M, N, c)
    lrms: numpy array with (m, n, C)
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    M, N, c = pan.shape
    m, n, C = lrms.shape
    ratio = int(np.round(M / m))
    print('get sharpening ratio: ', ratio)
    assert int(np.round(M / m)) == int(np.round(N / n))
    training_size = 256

    # 定义损失函数字典
    loss_functions = {
        'mae': 'mae',
        'log': modified_log_sobel(),
        'mse': 'mse'

    }

    # 分割数据集
    (train_pan, train_lrms), \
        (val_pan, val_lrms), \
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=32,seed=35)
    print(f"Train set: {len(train_pan)} grids, PAN {train_pan.shape}, LRMS {train_lrms.shape}")
    print(f"Val set: {len(val_pan)} grids, PAN {val_pan.shape}, LRMS {val_lrms.shape}")
    print(f"Test set: {len(test_pan)} grids, PAN {test_pan.shape}, LRMS {test_lrms.shape}")

    # 数据准备
    def prepare_data(pan_grids, lrms_grids, ratio):
        """
        为训练和验证集降采样数据，HRMS使用原始LRMS
        """
        N = pan_grids.shape[0]
        used_pan, used_lrms, used_hrms = [], [], []
        for i in range(N):
            pan_grid = pan_grids[i]
            lrms_grid = lrms_grids[i]
            lrms_down, pan_down = downgrade_images(lrms_grid, pan_grid, ratio, sensor='WV2')
            used_pan.append(pan_down)
            used_lrms.append(lrms_down)
            used_hrms.append(lrms_grid)
        used_pan = np.array(used_pan, dtype='float16')
        used_lrms = np.array(used_lrms, dtype='float16')
        used_hrms = np.array(used_hrms, dtype='float16')
        return used_pan, used_lrms, used_hrms

    # 准备训练和验证数据（仅一次）
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio)
    print(f"Prepared data: Train PAN {train_pan.shape}, LRMS {train_lrms.shape}, HRMS {train_hrms.shape}")
    print(f"Prepared data: Val PAN {val_pan.shape}, LRMS {val_lrms.shape}, HRMS {val_hrms.shape}")

    # 学习率调度
    def lr_schedule(epoch):
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

    # 遍历损失函数
    all_results = {}
    for loss_name, loss_fn in loss_functions.items():
        print(f"\nTraining with loss: {loss_name}\n")
        model_name = f'CMFNet_model_{loss_name}'
        results = {
            'train_metrics': {'PSNR': []},
            'val_metrics': {'PSNR': [], 'SSIM': [], 'SAM': [], 'ERGAS': [], 'Q8': [], 'SCC': []},
            'test_metrics': {'QNR': [], 'D_lambda': [], 'D_s': []},
            'time': []
        }

        # 创建和编译模型
        model = cmfnet(lrms_size=(16, 16, C), pan_size=(64, 64, c))
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=loss_fn, metrics=[psnr])
        checkpoint = ModelCheckpoint(f'./weightsWV2/{model_name}.h5',
                                    monitor='val_psnr', mode='max', verbose=1, save_best_only=True)

        # 训练
        train_start = time.time()
        history = model.fit(
            x=[train_lrms, train_pan],
            y=train_hrms,
            validation_data=([val_lrms, val_pan], val_hrms),
            batch_size=16,
            epochs=epoch,
            verbose=1,
            callbacks=[lr_scheduler, checkpoint]
        )
        train_end = time.time()
        results['train_metrics']['PSNR'].append(history.history['psnr'][-1])
        results['time'].append(train_end - train_start)

        # 验证集评估
        val_pred = model.predict([val_lrms, val_pan])
        val_metrics = evaluate_pansharpening(y_true=val_hrms, y_pred=val_pred, ratio=ratio)
        for metric_name, value in val_metrics.items():
            results['val_metrics'][metric_name].append(value)

        # 测试集评估（全尺度）
        model = cmfnet(lrms_size=(64, 64, C), pan_size=(256, 256, c))
        model.load_weights(f'./weightsWV2/{model_name}.h5')
        test_pred = model.predict([test_lrms, test_pan])
        test_pred = np.clip(test_pred, 0, 1)
        test_pred = np.uint8(test_pred * 255)

        # 计算 QNR
        qnr_values, d_lambda_values, d_s_values = [], [], []
        for i in range(len(test_pan)):
            qnr, d_lambda, d_s = QNR(test_lrms[i], test_pred[i], test_pan[i])
            qnr_values.append(qnr)
            d_lambda_values.append(d_lambda)
            d_s_values.append(d_s)
        results['test_metrics']['QNR'].append(np.mean(qnr_values))
        results['test_metrics']['D_lambda'].append(np.mean(d_lambda_values))
        results['test_metrics']['D_s'].append(np.mean(d_s_values))

        # 保存日志
        save_dir = './CMFNet_results/'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        log_file = os.path.join(save_dir, f"WV2_{model_name}_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"\nCMFNet Summary for {loss_name}:\n")
            f.write("-" * 50 + "\n")
            f.write("Training Metrics:\n")
            for metric, value in results['train_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nValidation Metrics:\n")
            for metric, value in results['val_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            f.write("\nTest Metrics:\n")
            for metric, value in results['test_metrics'].items():
                f.write(f"  {metric}: {value[-1]:.4f}\n")
            total_time = sum(results['time'])
            f.write(f"\nTotal Training Time: {total_time:.2f} sec\n")
            f.write("=" * 50 + "\n")

        # 打印结果
        print(f"\nCMFNet Summary for {loss_name}:\n")
        print("-" * 50 + "\n")
        print("Training Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        print("\nTest Metrics:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value[-1]:.4f}")
        total_time = sum(results['time'])
        print(f"\nTotal Training Time: {total_time:.2f} sec\n")
        print("=" * 50 + "\n")

        # 存储结果
        all_results[loss_name] = results

    return all_results

############ WV2
file_path = r"D:\桌面\对数损失\EGLL\imgWV2.mat"
mat_data = sio.loadmat(file_path)
used_ms = mat_data['I_MS']           # (320, 320, 8)
used_pan = mat_data['I_PAN']         # (1280, 1280)
RGB_indexes = mat_data['RGB_indexes']  # (1, 3)——5,3,2
used_pan = np.expand_dims(used_pan, -1) #(1280,1280,1)
'''normalization'''
max_patch, min_patch = np.max(used_ms, axis=(0,1)), np.min(used_ms, axis=(0,1))
used_ms = np.float32(used_ms-min_patch) / (max_patch - min_patch)
max_patch, min_patch = np.max(used_pan, axis=(0,1)), np.min(used_pan, axis=(0,1))
used_pan = np.float32(used_pan-min_patch) / (max_patch - min_patch)
print('ms shape: ', used_ms.shape, 'pan shape: ', used_pan.shape)

PanNet(used_pan[:,:,:], used_ms[:,:,:], epoch=2)
# LAGConv(used_pan[:,:,:], used_ms[:,:,:], epoch=100)
# CMFNet(used_pan[:,:,:], used_ms[:,:,:], epoch=100)
# PNN(used_pan[:,:,:], used_ms[:,:,:],epoch=100)
# FusionNet(used_pan[:,:,:], used_ms[:,:,:], epoch=100)

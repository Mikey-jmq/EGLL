import numpy as np
from keras.layers import Concatenate, Conv2D, Input, Layer, Add, Activation, BatchNormalization,Lambda
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
from evaluation import *

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

def split_dataset(pan, lrms, size=256, train_ratio=0.8, val_ratio=0.1, stride=32):
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
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
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

    return (train_pan_grids, train_lrms_grids), \
       (val_pan_grids, val_lrms_grids), \
       (test_pan_grids, test_lrms_grids)

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

def basic_unit(inputs, input_channels, mid_channels, out_channels,  name='basic'):
    x = Conv2D(mid_channels, (3, 3), padding='same', name=name + '_conv1')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(out_channels, (3, 3), padding='same', name=name + '_conv2')(x)
    return x

def lr_block(HR, LR, n_feat, name='lrblock'):
    # HR: HR image (B,H, W, C)
    # LR: LR MS image (B,h, w, C)
    M,N,C = HR.shape[-3:]
    m,n,_ = LR.shape[-3:]
    lr_hat = basic_unit(HR, C, n_feat, C, name=name + '_get_lr')
    lr_hat = resize1(LR.shape[-3:])(lr_hat)
    lr_residual = LR - lr_hat
    hr_residual = basic_unit(lr_residual, C, n_feat, C, name=name + '_get_hr_residual')
    hr_residual = resize1(HR.shape[-3:])(hr_residual)
    hr = Add()([HR, hr_residual])
    hr = basic_unit(hr, C, n_feat, C, name=name + '_prox')
    return hr

def pan_block(HR, PAN,  n_feat, name='panblock'):
    # inputs: HR image (B,H, W, ms_channels)
    # pan_input: PAN image (B,H, W, pan_channels)
    M,N,C = HR.shape[-3:]
    _,_,c = PAN.shape[-3:]
    pan_hat = basic_unit(HR, C, n_feat, c, name=name + '_get_pan')
    pan_residual = PAN - pan_hat
    hr_residual = basic_unit(pan_residual, c, n_feat, C, name=name + '_get_hr_residual')
    hr = Add()([HR, hr_residual])
    hr = basic_unit(hr, C, n_feat, C, name=name + '_prox')

    return hr

def gppnn(lrms_size=(16, 16, 3), pan_size = (64, 64, 1)):
    # 输入LRMS和PAN图像
    lrms_inputs = Input(lrms_size)
    pan_inputs = Input(pan_size)
    hr = resize1(pan_size)(lrms_inputs)
    for i in range(4):
        hr = lr_block(hr, lrms_inputs,64, name=f'lrblock_{i + 1}')
        hr = pan_block(hr, pan_inputs, 64,  name=f'panblock_{i + 1}')
    return Model(inputs=[lrms_inputs, pan_inputs], outputs=hr)

def GPPNN(pan, lrms, data='WV2', sensor=None, epoch=20, stride=32):
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
        (test_pan, test_lrms) = split_dataset(pan, lrms, size=training_size, stride=stride)
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
    train_pan, train_lrms, train_hrms = prepare_data(train_pan, train_lrms, ratio=ratio, sensor=sensor)
    val_pan, val_lrms, val_hrms = prepare_data(val_pan, val_lrms, ratio=ratio, sensor=sensor)
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
        checkpoint = ModelCheckpoint(f'./weights{data}/{model_name}.h5',
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
        model.load_weights(f'./weights{data}/{model_name}.h5')
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
        log_file = os.path.join(save_dir, f"{data}_{model_name}_log.txt")
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
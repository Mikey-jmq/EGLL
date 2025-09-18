import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
from tensorflow.keras import backend as K
from utils import downgrade_images
import cv2

#归一化
def min_max_normalize(img):
    """
    Min-max normalization for 2D (H, W), 3D (H, W, C) or 4D (B, H, W, C) arrays.
    Normalizes each channel separately.
    Args:
        img: Input array to normalize
    Returns:
        Normalized array with the same shape as input
    """
    if img.ndim == 2:
        # 2D case (H, W) - treat as single channel
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val == min_val:  # avoid division by zero
            return np.zeros_like(img)
        return (img - min_val) / (max_val - min_val)

    elif img.ndim == 3:
        # 3D case (H, W, C) - normalize each channel separately
        min_vals = np.min(img, axis=(0, 1))
        max_vals = np.max(img, axis=(0, 1))
        # Avoid division by zero for constant channels
        safe_diff = np.where(max_vals == min_vals, 1.0, max_vals - min_vals)
        return (img - min_vals) / safe_diff
    elif img.ndim == 4:
        # 4D case (B, H, W, C) - normalize each channel in each image separately
        min_vals = np.min(img, axis=(1, 2))  # shape (B, C)
        max_vals = np.max(img, axis=(1, 2))  # shape (B, C)
        min_vals = min_vals[:, np.newaxis, np.newaxis, :]
        max_vals = max_vals[:, np.newaxis, np.newaxis, :]

        # Avoid division by zero for constant channels
        safe_diff = np.where(max_vals == min_vals, 1.0, max_vals - min_vals)
        return (img - min_vals) / safe_diff

    else:
        raise ValueError(f"Unsupported array dimension: {img.ndim}. Expected 2D, 3D or 4D array.")
#PSNR
def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio averaged over samples and channels.
       输入必须归一化
       """
    # 确保输入是 float32 类型，避免类型不匹配
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    mse = K.mean(K.square(y_true * 255.0 - y_pred * 255.0), axis=(-3, -2, -1))
    return K.mean(20 * K.log(255.0 / K.sqrt(mse)) / np.log(10))

def upsample_lrms(lrms, ratio):
    """上采样LRMS到PAN的尺寸（简单双线性插值）"""
    return tf.image.resize(lrms, (lrms.shape[0]*ratio, lrms.shape[1]*ratio), method='bilinear').numpy()
#SCC
def SCC(y_true, y_pred):
    """SCC for 2D (H, W), 3D (H, W, C) or 4D (B, H, W, C) image; uint or float[0, 1]"""
    if not y_true.shape == y_pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Convert to NumPy arrays for NumPy operations
    y_true = y_true.numpy() if isinstance(y_true, tf.Tensor) else np.asarray(y_true)
    y_pred = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else np.asarray(y_pred)
    if y_true.ndim == 2:
        # 2D case (H, W)
        return np.corrcoef(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0, 1]
    elif y_true.ndim == 3:
        # 3D case (H, W, C)
        ccs = [np.corrcoef(y_true[..., i].reshape(1, -1), y_pred[..., i].reshape(1, -1))[0, 1]
               for i in range(y_true.shape[2])]
        return np.mean(ccs)
    elif y_true.ndim == 4:
        # 4D case (B, H, W, C)
        batch_ccs = []
        for b in range(y_true.shape[0]):
            # Process each image in the batch
            if y_true.shape[3] == 1:  # Single-channel case
                cc = np.corrcoef(y_true[b, ..., 0].reshape(1, -1),y_pred[b, ..., 0].reshape(1, -1))[0, 1]
            else:  # Multi-channel case
                ccs = [np.corrcoef(y_true[b, ..., i].reshape(1, -1),
                                   y_pred[b, ..., i].reshape(1, -1))[0, 1]
                       for i in range(y_true.shape[3])]
                cc = np.mean(ccs)
            batch_ccs.append(cc)
        return np.mean(batch_ccs)  # Return average across batch

    else:
        raise ValueError('Wrong input image dimensions. Expected 2D, 3D or 4D array.')

#SAM
def SAM(y_true, y_pred):
    """适合3D/4D的输入，输入可归一化可不归一化"""
    if len(y_true.shape) == 4:
        batch_size = tf.shape(y_true)[0]
        batch_indices = tf.range(batch_size)
        sam_values = tf.map_fn(lambda i: SAM(y_true[i], y_pred[i]),batch_indices,dtype=tf.float32)
        return tf.reduce_mean(sam_values)
    assert y_true.ndim == 3 and y_true.shape == y_pred.shape and y_true.shape[2]>1
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    dot_sum = np.sum(y_true * y_pred, axis=2)
    norm_true = np.sqrt((y_true**2).sum(axis=2))
    norm_pred = np.sqrt((y_pred**2).sum(axis=2))
    cos_theta = (dot_sum/(norm_true*norm_pred+np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))

#ERGAS
def ergas(y_true, y_pred, ratio=4):
    """ERGAS（相对全局无量纲综合误差）。
    参数：
       y_true: 形状为 (H,W,C) 或 (B,H,W,C) 的真实图像。
       y_pred: 形状为 (H,W,C) 或 (B,H,W,C) 的预测图像。
       ratio: 高分辨率与低分辨率的空间分辨率比（默认：4）。
    返回：
       ERGAS 值（float32），对于批量输入返回平均值。
    """
    if len(y_true.shape) == 4:
        batch_size = y_true.shape[0]
        batch_indices = tf.range(batch_size)
        ergas_values = tf.map_fn(lambda i: ergas(y_true[i], y_pred[i], ratio), batch_indices, dtype=tf.float32)
        return tf.reduce_mean(ergas_values)
    assert y_true.ndim == 3 and y_true.shape == y_pred.shape
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rmse2 = tf.reduce_mean(tf.square(y_true - y_pred), axis=[0, 1])
    mean_true = tf.reduce_mean(y_true, axis=[0, 1])
    relative_error_squared = rmse2 / (tf.square(mean_true) + tf.keras.backend.epsilon())
    return 100 / ratio * tf.sqrt(tf.reduce_mean(relative_error_squared))
#Q
def _q_index(y_true, y_pred, block_size=8):
    """
    Q-index for 2D single-channel image with shape (H, W).
    Args:
        y_true: Ground truth image, shape (H, W), float32 in [0, 1].
        y_pred: Predicted image, shape (H, W), float32 in [0, 1].
        block_size: Size of the local window (default: 8).
    Returns:
        Q-index value (scalar).
    """
    if not y_true.shape == y_pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    if len(y_true.shape) != 2:
        raise ValueError('Input shape must be (H, W).')
    if block_size < 2:
        raise ValueError('block_size should be greater than 1!')

    y_true = np.asarray(y_true, dtype=np.float64) if not isinstance(y_true, np.ndarray) else y_true.astype(np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64) if not isinstance(y_pred, np.ndarray) else y_pred.astype(np.float64)
    window = np.ones((block_size, block_size), dtype=np.float64) / (block_size ** 2)
    # 计算边界裁剪
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    slice_idx = slice(pad_topleft, -pad_bottomright)
    # 计算局部均值
    mu1 = cv2.filter2D(y_true, -1, window)[slice_idx, slice_idx]
    mu2 = cv2.filter2D(y_pred, -1, window)[slice_idx, slice_idx]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = cv2.filter2D(y_true ** 2, -1, window)[slice_idx, slice_idx] - mu1_sq
    sigma2_sq = cv2.filter2D(y_pred ** 2, -1, window)[slice_idx, slice_idx] - mu2_sq
    sigma12 = cv2.filter2D(y_true * y_pred, -1, window)[slice_idx, slice_idx] - mu1_mu2

    # 初始化 Q-index 映射
    qindex_map = np.ones_like(sigma12, dtype=np.float64)
    eps = np.finfo(np.float64).eps * 1e3

    # 情况 1: sigma == 0, mu != 0
    idx1 = ((sigma1_sq + sigma2_sq) < 1e-8) & ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx1] = 2 * mu1_mu2[idx1] / (mu1_sq + mu2_sq)[idx1] + eps

    # 情况 2: sigma != 0, mu == 0
    idx2 = ((sigma1_sq + sigma2_sq) > 1e-8) & ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx2] = 2 * sigma12[idx2] / (sigma1_sq + sigma2_sq)[idx2] + eps

    # 情况 3: sigma != 0, mu != 0
    idx3 = ((sigma1_sq + sigma2_sq) > 1e-8) & ((mu1_sq + mu2_sq) > 1e-8)
    qindex_map[idx3] = (2 * mu1_mu2[idx3] * 2 * sigma12[idx3]) / (
        (mu1_sq + mu2_sq)[idx3] * (sigma1_sq + sigma2_sq)[idx3] + eps)

    return np.mean(qindex_map)
def Q_index(y_true, y_pred, block_size=8):
    """
    Q-index for 2D (H,W), 3D (H,W,C), or 4D (B,H,W,C) images.
    Args:
        y_true: Ground truth image, shape (H,W), (H,W,C), or (B,H,W,C), float32 in [0, 1].
        y_pred: Predicted image, same shape as y_true, float32 in [0, 1].
        block_size: Size of the local window (default: 8).
    Returns:
        Q-index value (scalar), averaged over batch and channels.
    """
    if not y_true.shape == y_pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    if block_size < 2:
        raise ValueError('block_size should be greater than 1!')
    ndim = len(y_true.shape)
    if ndim not in [2, 3, 4]:
        raise ValueError('Input shape must be (H,W), (H,W,C), or (B,H,W,C).')
    # 2D 输入
    if ndim == 2:
        return _q_index(y_true, y_pred, block_size)
    # 3D 输入
    elif ndim == 3:
        channels = tf.shape(y_true)[2]
        if channels < 1:
            raise ValueError('3D input must have at least one channel.')
        # 逐通道计算 Q-index
        q_values = [
            _q_index(y_true[..., c], y_pred[..., c], block_size)
            for c in range(channels)
        ]
        return tf.reduce_mean(q_values)
    # 4D 输入
    else:
        channels = tf.shape(y_true)[3]
        if channels < 1:
            raise ValueError('4D input must have at least one channel.')
        # 展平批次和通道，形状 (B*C, H, W)
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[1], tf.shape(y_true)[2]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
        # 批量计算 Q-index
        q_values = tf.map_fn(
            lambda x: _q_index(x[0], x[1], block_size),
            (y_true_flat, y_pred_flat),
            dtype=tf.float64
        )

        return tf.reduce_mean(q_values)

#SSIM
def _ssim(y_true, y_pred, dynamic_range=1.0):
    """
    SSIM for 2D single-channel image with shape (H, W).
    Args:
        y_true: Ground truth image, shape (H, W), float32 in [0, 1].
        y_pred: Predicted image, shape (H, W), float32 in [0, 1].
        dynamic_range: Maximum pixel value (default: 1.0 for [0, 1] images).
    Returns:
        SSIM value (scalar).
    """
    if not y_true.shape == y_pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    if len(y_true.shape) != 2:
        raise ValueError('Input shape must be (H, W).')

    # 转换为 NumPy 数组并确保 float64
    y_true_np = y_true.numpy().astype(np.float64)
    y_pred_np = y_pred.numpy().astype(np.float64)
    # 定义稳定常数
    C1 = (0.01 * dynamic_range) ** 2
    C2 = (0.03 * dynamic_range) ** 2
    # 创建高斯核（11x11，标准差 1.5）
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    # 计算边界裁剪（模拟 VALID 卷积）
    pad = 5  # (11 - 1) / 2
    slice_idx = slice(pad, -pad)

    # 计算局部均值
    mu1 = cv2.filter2D(y_true_np, -1, window)[slice_idx, slice_idx]
    mu2 = cv2.filter2D(y_pred_np, -1, window)[slice_idx, slice_idx]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # 计算方差和协方差
    sigma1_sq = cv2.filter2D(y_true_np ** 2, -1, window)[slice_idx, slice_idx] - mu1_sq
    sigma2_sq = cv2.filter2D(y_pred_np ** 2, -1, window)[slice_idx, slice_idx] - mu2_sq
    sigma12 = cv2.filter2D(y_true_np * y_pred_np, -1, window)[slice_idx, slice_idx] - mu1_mu2

    # 计算 SSIM 映射
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)
def SSIM(y_true, y_pred, dynamic_range=1.0):
    """
    SSIM for 2D (H,W), 3D (H,W,C), or 4D (B,H,W,C) images.
    Args:
        y_true: Ground truth image, shape (H,W), (H,W,C), or (B,H,W,C), float32 in [0, 1].
        y_pred: Predicted image, same shape as y_true, float32 in [0, 1].
        dynamic_range: Maximum pixel value (default: 1.0 for [0, 1] images).
    Returns:
        SSIM value (scalar), averaged over batch and channels.
    """
    if not y_true.shape == y_pred.shape:
        raise ValueError('Input images must have the same dimensions.')
    ndim = len(y_true.shape)
    if ndim not in [2, 3, 4]:
        raise ValueError('Input shape must be (H,W), (H,W,C), or (B,H,W,C).')

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # 2D 输入
    if ndim == 2:
        return _ssim(y_true, y_pred, dynamic_range)
    # 3D 输入
    elif ndim == 3:
        channels = tf.shape(y_true)[2]
        if channels < 1:
            raise ValueError('3D input must have at least one channel.')
        ssim_vals = [
            _ssim(y_true[..., c], y_pred[..., c], dynamic_range)
            for c in range(channels)
        ]
        return tf.reduce_mean(ssim_vals)
    # 4D 输入
    else:
        channels = tf.shape(y_true)[3]
        if channels < 1:
            raise ValueError('4D input must have at least one channel.')
        y_true_flat = tf.reshape(y_true, [-1, tf.shape(y_true)[1], tf.shape(y_true)[2]])
        y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[1], tf.shape(y_pred)[2]])
        ssim_vals = tf.map_fn(
            lambda x: _ssim(x[0], x[1], dynamic_range),
            (y_true_flat, y_pred_flat),
            dtype=tf.float64
        )

        return tf.reduce_mean(ssim_vals)

#无参考指标QNR（先计算Dλ)
#计算单通道Q，方便计算D的
def D_lambda(lrms, fused_image, block_size=32, p=1):
    """
    计算光谱失真指标 D_lambda。
    Args:
        lrms: 低分辨率多光谱图像，形状 (h, w, C)，float32，归一化到 [0, 1]。
        fused_image: 融合后的高分辨率多光谱图像，形状 (H, W, C)，float32，未归一化。
        block_size: 局部窗口大小，默认为 32。
        p: 范数参数，默认为 1。
    Returns:
        D_lambda 值（标量）。
    """
    # 输入验证
    if lrms.ndim != 3 or fused_image.ndim != 3:
        raise ValueError("Inputs must have shape (H, W, C).")
    if lrms.shape[-1] != fused_image.shape[-1]:
        raise ValueError("lrms and fused_image must have the same number of channels.")

    # 转换类型并归一化 fused_image
    lrms = tf.cast(lrms, tf.float32)
    fused_image = tf.cast(fused_image, tf.float32)
    # 对 fused_image 逐通道归一化到 [0, 1]
    min_vals = tf.reduce_min(fused_image, axis=(0, 1), keepdims=True)
    max_vals = tf.reduce_max(fused_image, axis=(0, 1), keepdims=True)
    fused_image = (fused_image - min_vals) / (max_vals - min_vals + tf.keras.backend.epsilon())
    C = lrms.shape[-1]
    # 计算所有波段对的 Q 值
    Q_fake = []
    Q_lm = []
    for i in range(C):
        for j in range(i + 1, C):
            # fused_image 的波段对
            band1_fake = fused_image[..., i]  # 形状 (H, W)
            band2_fake = fused_image[..., j]  # 形状 (H, W)
            Q_fake.append(_q_index(band1_fake, band2_fake, block_size=block_size))
            # lrms_upsampled 的波段对
            band1_lm = lrms[..., i]  # 形状 (H, W) 或 (h, w)
            band2_lm = lrms[..., j]  # 形状 (H, W) 或 (h, w)
            Q_lm.append(_q_index(band1_lm, band2_lm, block_size=block_size))

    Q_fake = tf.stack(Q_fake)
    Q_lm = tf.stack(Q_lm)
    D_lambda_index = tf.reduce_mean(tf.abs(Q_fake - Q_lm) ** p) ** (1 / p)
    return D_lambda_index

def D_s(lrms, fused_image, pan, scale=4, block_size=32, q=1):
    """
    计算空间失真指标 D_s。
    Args:
        lrms: 低分辨率多光谱图像，形状 (h, w, C)，float32，归一化到 [0, 1]。
        fused_image: 融合后的高分辨率多光谱图像，形状 (H, W, C)，float32，未归一化。
        pan: 高分辨率全色图像，形状 (H, W, 1)，float32，归一化到 [0, 1]。
        scale: 分辨率比例（H/h），默认为 4。
        block_size: 局部窗口大小，默认为 32。
        q: 范数参数，默认为 1。
    Returns:
        D_s 值（标量）。
    """
    # 输入验证
    if lrms.ndim != 3 or fused_image.ndim != 3 or pan.ndim != 3:
        raise ValueError("Inputs must have shape (H, W, C) or (H, W, 1) for pan.")
    if lrms.shape[-1] != fused_image.shape[-1]:
        raise ValueError("lrms and fused_image must have the same number of channels.")
    if pan.shape[-1] != 1:
        raise ValueError("pan must have one channel.")
    if fused_image.shape[0] != pan.shape[0] or fused_image.shape[1] != pan.shape[1]:
        raise ValueError("fused_image and pan must have the same spatial dimensions.")
    if fused_image.shape[0] // lrms.shape[0] != scale:
        raise ValueError("Spatial resolution must be compatible with scale.")

    # 转换类型并归一化 fused_image
    lrms = tf.cast(lrms, tf.float32)
    fused_image = tf.cast(fused_image, tf.float32)
    pan = tf.cast(pan, tf.float32)
    min_vals = tf.reduce_min(fused_image, axis=(0, 1), keepdims=True)
    max_vals = tf.reduce_max(fused_image, axis=(0, 1), keepdims=True)
    fused_image = (fused_image - min_vals) / (max_vals - min_vals + tf.keras.backend.epsilon())
    # 降采样 pan 到 lrms 分辨率
    pan_lr = tf.image.resize(pan, [lrms.shape[0], lrms.shape[1]], method='bilinear')
    pan_lr = tf.squeeze(pan_lr,axis=-1)
    pan = tf.squeeze(pan,axis=-1)

    C = lrms.shape[-1]
    # 计算 Q 值
    Q_hr = []
    Q_lr = []
    for i in range(C):
        # HR: fused_image vs. pan
        band_fake = fused_image[..., i]
        Q_hr.append(_q_index(band_fake, pan, block_size=block_size))
        # LR: lrms vs. pan_lr
        band_lm = lrms[..., i]
        Q_lr.append(_q_index(band_lm, pan_lr, block_size=block_size))

    Q_hr = tf.stack(Q_hr)
    Q_lr = tf.stack(Q_lr)
    D_s_index = tf.reduce_mean(tf.abs(Q_hr - Q_lr) ** q) ** (1 / q)
    return D_s_index

def QNR(lrms, fused_image, pan, scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """
    计算无参考 QNR 指标。
    Args:
        lrms: 低分辨率多光谱图像，形状 (h, w, C)，float32，归一化到 [0, 1]。
        fused_image: 融合后的高分辨率多光谱图像，形状 (H, W, C)，float32，未归一化。
        pan: 高分辨率全色图像，形状 (H, W, 1)，float32，归一化到 [0, 1]。
        scale: 分辨率比例（H/h），默认为 4。
        block_size: 局部窗口大小，默认为 32。
        p: D_lambda 的范数参数，默认为 1。
        q: D_s 的范数参数，默认为 1。
        alpha: D_lambda 的权重，默认为 1。
        beta: D_s 的权重，默认为 1。
    Returns:
        QNR 值（标量），以及 D_lambda 和 D_s。
    """
    D_lambda_idx = D_lambda(lrms, fused_image, block_size, p)
    D_s_idx = D_s(lrms, fused_image, pan, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return QNR_idx, D_lambda_idx, D_s_idx

def evaluate_pansharpening(y_true, y_pred, ratio=4):
    """
    全色锐化综合评估
    输入:
        y_true: 参考HRMS图像 (B,H, W, 8)，范围[0, 1]
        y_pred: 融合结果 (B,H, W, 8)
        ratio: 分辨率比例（用于ERGAS）
    返回: 字典包含所有指标
    """
    # 转换为 float32 确保一致性
    y_true_float = y_true.astype(np.float32)
    y_pred_float = y_pred.astype(np.float32)

    # 转换为0-255范围（部分指标需要）
    y_true_uint = (y_true_float * 255).astype(np.uint8)
    y_pred_uint = (y_pred_float * 255).astype(np.uint8)

    metrics = {
        'PSNR': psnr(y_true, y_pred).numpy(),
        'SSIM': SSIM(y_true, y_pred),
        'SAM': SAM(y_true_uint, y_pred_uint),
        'ERGAS': ergas(y_true_uint, y_pred_uint, ratio),
        'Q8': Q_index(y_true, y_pred,block_size=8),
        'SCC':SCC(y_true,y_pred)
    }
    return metrics
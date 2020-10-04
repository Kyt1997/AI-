# -*- coding: utf-8 -*-
"""
Created on Fri Oct 2 15:02:44 2020

@author: 孔宇韬
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
import tensorflow as tf
from keras.models import Model   #非线性网路，采用Model
from keras.layers import Input,Dense,Flatten,Reshape
from keras.optimizers import Adam

def get_image_paths(dire):
    return [x.path for x in os.scandir(dire) if x.name.endswith('.jpg') or x.name.endswith('.png')]

A_PATH = input("请输入第一个图片集的绝对路径：")
B_PATH  = input("请输入第二个图片集的绝对路径：")
images_A = get_image_paths(A_PATH) #images_trump存放每一张图片的绝对路径
images_B = get_image_paths(B_PATH)
print('数据集A有{}张图片，数据集B有{}张图片'.format(len(images_A),len(images_B)))


def load_images(image_paths):
    iter_all_images = (cv2.imread(fn) for fn in image_paths)
    #创造一个存放图片的空间，顺便打上id序号
    for i,image in enumerate(iter_all_images):
        if i == 0:
            all_images = np.empty((len(image_paths),) + image.shape,dtype = image.dtype) #(len,shape[0],shape[1]，shape[2])的四维
        all_images[i] = image
    return all_images

A_images = load_images(images_A[0:3])
B_images = load_images(images_B[0:3])
print(A_images.shape)
print(A_images.shape)

def get_transpose_axes(n):
    if n % 2== 0:
        y_axes = list(range(1,n-1,2))
        x_axes = list(range(0,n-1,2))
    else:
        y_axes = list(range(0,n-1,2))
        x_axes = list(range(1,n-1,2))
    return y_axes,x_axes,[n-1]

def stack_images(images):  #吧多张图片拼合成一张大图片
    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    return np.transpose(images,axes = np.concatenate(new_axes)).reshape(new_shape)

print("初步观察数据集")
figure = np.concatenate([A_images,B_images],axis=0)
figure = figure.reshape((2,3)+figure.shape[1:])
figure = stack_images(figure)

plt.imshow(cv2.cvtColor(figure,cv2.COLOR_BGR2RGB))
plt.show()



from keras.utils import conv_utils
from keras.engine.topology import Layer
import keras.backend as K

#原来的函数不知道放在那哪了。。。所以网上找了一份源码用着
def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format



class PixelShuffler(Layer): 
    # 初始化 子像素卷积层，并在输入数据时，对数据进行标准化处理。
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):
        # 根据得到输入层图层 batch_size，h ，w，c 的大小
        input_shape = K.int_shape(inputs)
        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size

        # 计算转换后的图层大小与通道数
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        # 先将图层分开，并且将每一层装换到自己应该到维度
        # 最后再利用一次 reshape 函数（计算机会从外到里的一个个的将数据排下来），这就可以转成指定大小的图层了
        out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    # compute_output_shape()函数用来输出这一层输出尺寸的大小
    # 尺寸是根据input_shape以及我们定义的output_shape计算的。
    def compute_output_shape(self, input_shape):
        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1]  if input_shape[2] is not None else None
        channels = input_shape[3] // self.size[0] // self.size[1]

        return (input_shape[0],
                height,
                width,
                channels)

    # 设置配置文件
    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def conv(filters):
    def block(x):
        x = Conv2D(filters,kernel_size = 5,strides = 2,padding = 'same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale(filters):
    def block(x):
        x = Conv2D(filters*4,kernel_size =3,padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

# x1 = tf.convert_to_tensor(A_images,dtype=tf.float32)
# x2 = conv(126)(x1)
# x3 = upscale(3)(x2)
# print("将大小为 {} 的图片传入 filters 为 126 的下采样层中得到大小为 {} 的图层。".format(x1.shape, x2.shape))
# print("将大小为 {} 的图层传入 filters 为  3  的上采样层中得到大小为 {} 的图片。".format(x2.shape, x3.shape))

IMAGE_SHAPE = (64,64,3)
ENCODER_DIM = 1024 

def Encoder():   #先定义好编码器的网络结构，创建一个实例
    input_ = Input(shape = IMAGE_SHAPE) #创建一个输入图片规格大小的张量
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Dense(ENCODER_DIM)(Flatten()(x))
    x = Dense(4*4*1024)(x)
    x = Reshape((4,4,1024))(x)
    x = upscale(512)(x)
    return Model(input_,x)  #既是x也是y


def Decoder():
    input_  = Input(shape = (8,8,512))
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(3,kernel_size=5,padding='same',activation = 'sigmoid')(x)
    return Model(input_,x)
    
optimizer = Adam(lr = 5e-5,beta_1=0.5,beta_2=0.999)
encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)
autoencoder_A = Model(x,decoder_A(encoder(x)))  #定义好整个A的模型结构，先进入编码器再到解码器
autoencoder_B = Model(x,decoder_B(encoder(x))) #定义好整个B的模型结构

autoencoder_A.compile(optimizer=optimizer,loss='mean_absolute_error') #定义好优化器的参数
autoencoder_B.compile(optimizer=optimizer,loss='mean_absolute_error')
    


def random_transform(image):
    h,w = image.shape[0:2]
    #既要旋转也要缩放也要位移
    rotation = np.random.uniform(-10,10)
    scale = np.random.uniform(0.95,1.05)
    tx = np.random.uniform(-0.05,0.05) * w  #随机地产生一些平移距离
    ty = np.random.uniform(-0.05,0.05)* h
    
    mat = cv2.getRotationMatrix2D((w//2,h//2),rotation,scale) #返回一个仿射变换矩阵，参数分别是旋转中心，旋转的角度，缩放大小
    mat[:,2] += (tx,ty)
    
    result = cv2.warpAffine(image,mat,(w,h),borderMode=cv2.BORDER_REPLICATE)  #将仿射变换矩阵放入其中
    
    if np.random.random() < 0.4:
        result  = result[:,::-1]
    return result
    

#观察其变化
# old_image = A_images[1]
# transform_image = random_transform(old_image)
# print("变化前图片大小为{}\n变化后图片大小为{}".format(old_image.shape, transform_image.shape))
# figure = np.concatenate([old_image,transform_image],axis=0)
# figure = stack_images(figure)

# plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))



def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38). 下面的Eq 都分别对应着论文中的公式
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T



def random_warp(image):
    range_ = np.linspace(128-80,128+80,5)
    mapx = np.broadcast_to(range_,(5,5)) #复制五份
    mapy = mapx.T
    mapx = mapx + np.random.normal(size=(5,5),scale=5)
    
    mapy = mapy+ np.random.normal(size=(5,5),scale=5)
    #将中间部分裁剪成64*64的
    interp_mapx = cv2.resize(mapx,(80,80))[8:72,8:72].astype('float32')
    interp_mapy = cv2.resize(mapy,(80,80))[8:72,8:72].astype('float32')
    
    warped_image = cv2.remap(image,interp_mapx,interp_mapy,cv2.INTER_LINEAR)
    src_points = np.stack([mapx.ravel(),mapy.ravel()],axis=1)
    dst_points = np.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama(src_points,dst_points,True)[0:2] 
    target_image = cv2.warpAffine(image,mat,(64,64))
    
    return warped_image,target_image


# warped_image,target_image = random_warp(transform_image)
# print("warpe 前图片大小{}\nwarpe 后图片大小{}".format(
#     transform_image.shape, warped_image.shape))

def get_training_data(images,batch_size):
    indices = np.random.randint(len(images),size=batch_size)
    for i,index in enumerate(indices):
        image = images[index]
        image = random_transform(image)
        warped_img,target_img = random_warp(image)
        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape,warped_img.dtype)
            target_images = np.empty((batch_size,) +target_img.shape,warped_img.dtype)
            
        warped_images[i] = warped_img
        target_images[i] = target_img
    return warped_images,target_images

images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

#将数据进行分批，每个批次 20 条
# warped_A, target_A = get_training_data(images_A, 20)
# warped_A.shape, target_A.shape

def save_model_weights():
    E = input('请输入你要保存模型的路径：')
    
    encoder  .save_weights(E+"\encoder.h5")
    decoder_A.save_weights(E+"\decoder_A.h5")
    decoder_B.save_weights(E+"\decoder_B.h5")
    print("save model weights")
    return E

epochs = 10000
for epoch in range(epochs):
    print('当前是第{}轮迭代'.format(epoch))
    batch_size = 26
    warped_A,target_A = get_training_data(images_A, batch_size)
    warped_B,target_B = get_training_data(images_B, batch_size)
    loss_A = autoencoder_A.train_on_batch(warped_A,target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B,target_B)
    print('数据集A的loss是{}'.format(loss_A))
    print('数据集B的loss是{}'.format(loss_B))
    
E = save_model_weights()

'''_______________________________________________________________________________________'''
#这里开始测试我们的模型
print("开始加载模型，请耐心等待……")
encoder  .load_weights(E+"\encoder.h5")
decoder_A.load_weights(E+"\decoder_A.h5")
decoder_B.load_weights(E+"\decoder_B.h5")


# 下面代码和训练代码类似
# 获取图片，并对图片进行预处理
images_A = get_image_paths(A_PATH)
images_B = get_image_paths(B_PATH)
# 图片进行归一化处理
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
batch_size = 64
warped_A, target_A = get_training_data(images_A, batch_size)
warped_B, target_B = get_training_data(images_B, batch_size)


# 分别取当下批次下的数据集A和数据集B的图片的前三张进行观察
test_A = target_A[0:3]
test_B = target_B[0:3]

print("开始预测，请耐心等待……")
# 进行拼接 原图 A - 解码器 A 生成的图 - 解码器 B 生成的图
figure_A = np.stack([
    test_A,
    autoencoder_A.predict(test_A),
    autoencoder_B.predict(test_A),
], axis=1)
# 进行拼接  原图 B - 解码器 B 生成的图 - 解码器 A 生成的图
figure_B = np.stack([
    test_B,
    autoencoder_B.predict(test_B),
    autoencoder_A.predict(test_B),
], axis=1)

print("开始画图，请耐心等待……")
# 将多幅图拼成一幅图 （已在数据可视化部分进行了详细讲解）
figure = np.concatenate([figure_A, figure_B], axis=0)
figure = figure.reshape((2, 3) + figure.shape[1:])
figure = stack_images(figure)

# 将图片进行反归一化
figure = np.clip(figure * 255, 0, 255).astype('uint8')

# 显示图片
plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))
plt.show()

print('呈现出两个数据集的各前三张图，第一列是原图，第二列是数据集A的解码器生成的图，第三列是另一个数据集B的解码器生成的图,以此类推')
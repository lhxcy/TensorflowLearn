"""
Basic Multi GPU computation example using TensorFlow library.
"""
'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''
"""
tf.ConfigProto一般用在创建session的时候。用来对session进行参数配置
with tf.Session(config = tf.ConfigProto(...),...)


1:tensorflow ConfigProo

#tf.ConfigProto()的参数
log_device_placement=True : 是否打印设备分配日志
allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)


2:控制GPU资源使用率

#allow growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
# 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
#内存，所以会导致碎片

# per_process_gpu_memory_fraction
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config, ...)
#设置每个GPU应该拿出多少容量给进程使用，0.4代表 40%


3:控制使用哪块GPU
~/ CUDA_VISIBLE_DEVICES=0  python your.py#使用GPU0
~/ CUDA_VISIBLE_DEVICES=0,1 python your.py#使用GPU0,1
#注意单词不要打错

#或者在 程序开头
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1

"""


import numpy as np
import tensorflow as tf
import datetime

#Processing Units logs
log_device_placement = True

#Num of multiplications tp perform
n = 10

'''
Example: compute A^n + B^n on 2 GPUs
Results on 8 cores with 2 GTX-980:
 * Single GPU computation time: 0:00:11.277449
 * Multi GPU computation time: 0:00:07.131701
'''

# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device("/cpu:0"):
	#Addition of all elements in c1, i.e. A^n + B^n
	sum = tf.add_n(c1)
# tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，等
t1_1 =datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placment=log_device_placement)) as sess:
	sess.run(sum,feed_dict={a:A,b:B})
t2_1 = datetime.datetime.now()

'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # Compute B^n and store result in c2
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))

























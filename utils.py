#-*- coding: utf-8 -*-
"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import scipy as sp
import scipy.io
import numpy as np
from glob import glob
import sys
import pickle
from time import gmtime, strftime
#from six.moves import xrange
import matplotlib.pyplot as plt
import lmdb
import os, gzip
from scipy.io import loadmat

import tensorflow as tf
import tensorflow.contrib.slim as slim

img_path = "img"
sound_path = "LMS"
img_path_test = "validation/img"
sound_path_test = "validation/LMS"
input_fname_pattern = '*.jpg'
input_sounds_pattern = '*.png'
data_path = "/media/media/9EA2104DA2102BF1/creat-TwistedW/CMAV/data"
seed_num = 548

def load_Sub(dataset_name, n_classes):
    L1 = glob(os.path.join(data_path, dataset_name, img_path, "bassoon", input_fname_pattern))
    L2 = glob(os.path.join(data_path, dataset_name, img_path, "cello", input_fname_pattern))
    L3 = glob(os.path.join(data_path, dataset_name, img_path, "clarinet", input_fname_pattern))
    L4 = glob(os.path.join(data_path, dataset_name, img_path, "double_bass", input_fname_pattern))
    L5 = glob(os.path.join(data_path, dataset_name, img_path, "flute", input_fname_pattern))
    L6 = glob(os.path.join(data_path, dataset_name, img_path, "horn", input_fname_pattern))
    L7 = glob(os.path.join(data_path, dataset_name, img_path, "oboe", input_fname_pattern))
    L8 = glob(os.path.join(data_path, dataset_name, img_path, "sax", input_fname_pattern))
    L9 = glob(os.path.join(data_path, dataset_name, img_path, "trombone", input_fname_pattern))
    L10 = glob(os.path.join(data_path, dataset_name, img_path, "trumpet", input_fname_pattern))
    L11 = glob(os.path.join(data_path, dataset_name, img_path, "tuba", input_fname_pattern))
    L12 = glob(os.path.join(data_path, dataset_name, img_path, "viola", input_fname_pattern))
    L13 = glob(os.path.join(data_path, dataset_name, img_path, "violin", input_fname_pattern))

    data_X = np.concatenate((L1, L2), axis=0)
    data_X = np.concatenate((data_X, L3), axis=0)
    data_X = np.concatenate((data_X, L4), axis=0)
    data_X = np.concatenate((data_X, L5), axis=0)
    data_X = np.concatenate((data_X, L6), axis=0)
    data_X = np.concatenate((data_X, L7), axis=0)
    data_X = np.concatenate((data_X, L8), axis=0)
    data_X = np.concatenate((data_X, L9), axis=0)
    data_X = np.concatenate((data_X, L10), axis=0)
    data_X = np.concatenate((data_X, L11), axis=0)
    data_X = np.concatenate((data_X, L12), axis=0)
    data_X = np.concatenate((data_X, L13), axis=0)

    data_mis_X = np.concatenate((L5, L7), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L2), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L1), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L10), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L8), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L13), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L3), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L11), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L9), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L6), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L4), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L12), axis=0)

    S1 = glob(os.path.join(data_path, dataset_name, sound_path, "bassoon_lms", input_sounds_pattern))
    S2 = glob(os.path.join(data_path, dataset_name, sound_path, "cello_lms", input_sounds_pattern))
    S3 = glob(os.path.join(data_path, dataset_name, sound_path, "clarinet_lms", input_sounds_pattern))
    S4 = glob(os.path.join(data_path, dataset_name, sound_path, "double_bass_lms", input_sounds_pattern))
    S5 = glob(os.path.join(data_path, dataset_name, sound_path, "flute_lms", input_sounds_pattern))
    S6 = glob(os.path.join(data_path, dataset_name, sound_path, "horn_lms", input_sounds_pattern))
    S7 = glob(os.path.join(data_path, dataset_name, sound_path, "oboe_lms", input_sounds_pattern))
    S8 = glob(os.path.join(data_path, dataset_name, sound_path, "sax_lms", input_sounds_pattern))
    S9 = glob(os.path.join(data_path, dataset_name, sound_path, "trombone_lms", input_sounds_pattern))
    S10 = glob(os.path.join(data_path, dataset_name, sound_path, "trumpet_lms", input_sounds_pattern))
    S11 = glob(os.path.join(data_path, dataset_name, sound_path, "tuba_lms", input_sounds_pattern))
    S12 = glob(os.path.join(data_path, dataset_name, sound_path, "viola_lms", input_sounds_pattern))
    S13 = glob(os.path.join(data_path, dataset_name, sound_path, "violin_lms", input_sounds_pattern))

    data_s = np.concatenate((S1, S2), axis=0)
    data_s = np.concatenate((data_s, S3), axis=0)
    data_s = np.concatenate((data_s, S4), axis=0)
    data_s = np.concatenate((data_s, S5), axis=0)
    data_s = np.concatenate((data_s, S6), axis=0)
    data_s = np.concatenate((data_s, S7), axis=0)
    data_s = np.concatenate((data_s, S8), axis=0)
    data_s = np.concatenate((data_s, S9), axis=0)
    data_s = np.concatenate((data_s, S10), axis=0)
    data_s = np.concatenate((data_s, S11), axis=0)
    data_s = np.concatenate((data_s, S12), axis=0)
    data_s = np.concatenate((data_s, S13), axis=0)

    data_mis_s = np.concatenate((S5, S7), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S2), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S1), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S10), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S8), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S13), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S3), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S11), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S9), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S6), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S4), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S12), axis=0)

    ground_truths = []

    for _ in range(len(L1)):
        label_index = 0
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L2)):
        label_index = 1
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L3)):
        label_index = 2
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L4)):
        label_index = 3
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L5)):
        label_index = 4
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L6)):
        label_index = 5
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L7)):
        label_index = 6
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L8)):
        label_index = 7
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L9)):
        label_index = 8
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L10)):
        label_index = 9
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L11)):
        label_index = 10
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L12)):
        label_index = 11
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L13)):
        label_index = 12
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)

    seed = seed_num
    np.random.seed(seed)  # 确保每次生成的随机数相同
    np.random.shuffle(data_X) # 将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(data_mis_X)
    np.random.seed(seed)
    np.random.shuffle(data_s)
    np.random.seed(seed)
    np.random.shuffle(data_mis_s)
    np.random.seed(seed)
    np.random.shuffle(ground_truths)
    return data_X, data_s, data_mis_X, data_mis_s, ground_truths

def load_Sub_test(dataset_name, n_classes):
    L1 = glob(os.path.join(data_path, dataset_name, img_path_test, "bassoon", input_fname_pattern))
    L2 = glob(os.path.join(data_path, dataset_name, img_path_test, "cello", input_fname_pattern))
    L3 = glob(os.path.join(data_path, dataset_name, img_path_test, "clarinet", input_fname_pattern))
    L4 = glob(os.path.join(data_path, dataset_name, img_path_test, "double_bass", input_fname_pattern))
    L5 = glob(os.path.join(data_path, dataset_name, img_path_test, "flute", input_fname_pattern))
    L6 = glob(os.path.join(data_path, dataset_name, img_path_test, "horn", input_fname_pattern))
    L7 = glob(os.path.join(data_path, dataset_name, img_path_test, "oboe", input_fname_pattern))
    L8 = glob(os.path.join(data_path, dataset_name, img_path_test, "sax", input_fname_pattern))
    L9 = glob(os.path.join(data_path, dataset_name, img_path_test, "trombone", input_fname_pattern))
    L10 = glob(os.path.join(data_path, dataset_name, img_path_test, "trumpet", input_fname_pattern))
    L11 = glob(os.path.join(data_path, dataset_name, img_path_test, "tuba", input_fname_pattern))
    L12 = glob(os.path.join(data_path, dataset_name, img_path_test, "viola", input_fname_pattern))
    L13 = glob(os.path.join(data_path, dataset_name, img_path_test, "violin", input_fname_pattern))

    data_X = np.concatenate((L1, L2), axis=0)
    data_X = np.concatenate((data_X, L3), axis=0)
    data_X = np.concatenate((data_X, L4), axis=0)
    data_X = np.concatenate((data_X, L5), axis=0)
    data_X = np.concatenate((data_X, L6), axis=0)
    data_X = np.concatenate((data_X, L7), axis=0)
    data_X = np.concatenate((data_X, L8), axis=0)
    data_X = np.concatenate((data_X, L9), axis=0)
    data_X = np.concatenate((data_X, L10), axis=0)
    data_X = np.concatenate((data_X, L11), axis=0)
    data_X = np.concatenate((data_X, L12), axis=0)
    data_X = np.concatenate((data_X, L13), axis=0)

    data_mis_X = np.concatenate((L5, L7), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L2), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L1), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L10), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L8), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L13), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L3), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L11), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L9), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L6), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L4), axis=0)
    data_mis_X = np.concatenate((data_mis_X, L12), axis=0)

    S1 = glob(os.path.join(data_path, dataset_name, sound_path_test, "bassoon", input_sounds_pattern))
    S2 = glob(os.path.join(data_path, dataset_name, sound_path_test, "cello", input_sounds_pattern))
    S3 = glob(os.path.join(data_path, dataset_name, sound_path_test, "clarinet", input_sounds_pattern))
    S4 = glob(os.path.join(data_path, dataset_name, sound_path_test, "double_bass", input_sounds_pattern))
    S5 = glob(os.path.join(data_path, dataset_name, sound_path_test, "flute", input_sounds_pattern))
    S6 = glob(os.path.join(data_path, dataset_name, sound_path_test, "horn", input_sounds_pattern))
    S7 = glob(os.path.join(data_path, dataset_name, sound_path_test, "oboe", input_sounds_pattern))
    S8 = glob(os.path.join(data_path, dataset_name, sound_path_test, "sax", input_sounds_pattern))
    S9 = glob(os.path.join(data_path, dataset_name, sound_path_test, "trombone", input_sounds_pattern))
    S10 = glob(os.path.join(data_path, dataset_name, sound_path_test, "trumpet", input_sounds_pattern))
    S11 = glob(os.path.join(data_path, dataset_name, sound_path_test, "tuba", input_sounds_pattern))
    S12 = glob(os.path.join(data_path, dataset_name, sound_path_test, "viola", input_sounds_pattern))
    S13 = glob(os.path.join(data_path, dataset_name, sound_path_test, "violin", input_sounds_pattern))

    data_s = np.concatenate((S1, S2), axis=0)
    data_s = np.concatenate((data_s, S3), axis=0)
    data_s = np.concatenate((data_s, S4), axis=0)
    data_s = np.concatenate((data_s, S5), axis=0)
    data_s = np.concatenate((data_s, S6), axis=0)
    data_s = np.concatenate((data_s, S7), axis=0)
    data_s = np.concatenate((data_s, S8), axis=0)
    data_s = np.concatenate((data_s, S9), axis=0)
    data_s = np.concatenate((data_s, S10), axis=0)
    data_s = np.concatenate((data_s, S11), axis=0)
    data_s = np.concatenate((data_s, S12), axis=0)
    data_s = np.concatenate((data_s, S13), axis=0)

    data_mis_s = np.concatenate((S5, S7), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S2), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S1), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S10), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S8), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S13), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S3), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S11), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S9), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S6), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S4), axis=0)
    data_mis_s = np.concatenate((data_mis_s, S12), axis=0)

    ground_truths = []

    for _ in range(len(L1)):
        label_index = 0
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L2)):
        label_index = 1
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L3)):
        label_index = 2
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L4)):
        label_index = 3
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L5)):
        label_index = 4
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L6)):
        label_index = 5
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L7)):
        label_index = 6
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L8)):
        label_index = 7
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L9)):
        label_index = 8
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L10)):
        label_index = 9
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L11)):
        label_index = 10
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L12)):
        label_index = 11
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)
    for _ in range(len(L13)):
        label_index = 12
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        ground_truths.append(ground_truth)

    seed = seed_num
    np.random.seed(seed)  # 确保每次生成的随机数相同
    np.random.shuffle(data_X) # 将mnist数据集中数据的位置打乱
    np.random.seed(seed)
    np.random.shuffle(data_mis_X)
    np.random.seed(seed)
    np.random.shuffle(data_s)
    np.random.seed(seed)
    np.random.shuffle(data_mis_s)
    np.random.seed(seed)
    np.random.shuffle(ground_truths)
    return data_X, data_s, data_mis_X, data_mis_s, ground_truths


def load_batch(fpath):
    with open(fpath, 'rb') as f:
        if sys.version_info > (3, 0):
            # Python3
            d = pickle.load(f, encoding='latin1')
        else:
            # Python2
            d = pickle.load(f)
    data = d["data"]
    labels = d["labels"]
    return data, labels

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

#显示所有变量的tensor类型
def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    # 此处的np.argmax(id, 1)是用来判断此处的类别到底是几，如np.argmax([[0,0,1,0,0,0,0,0,0,0]],1)=2,输出最大的数所在的第二维度数字
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)
    plt.close()

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def get_pix_image(Image, random_index, ih=108, iw=130, oh=64, ow=64):
    batch_images_files = Image[random_index]
    batch_I = [
        get_image(batch_file,
                  input_height=ih,
                  input_width=iw,
                  resize_height=oh,
                  resize_width=ow,
                  ) for batch_file in batch_images_files]

    batch_images = np.array(batch_I).astype(np.float32)
    return batch_images

def get_sound(Sound, random_index, ih=0, iw=0, oh=64, ow=64):
    batch_sounds_files = Sound[random_index]
    batch_S = [
        get_image(batch_file_s,
                  input_height=0,
                  input_width=0,
                  resize_height=oh,
                  resize_width=ow,
                  crop=False
                  ) for batch_file_s in batch_sounds_files]

    batch_sounds = np.array(batch_S).astype(np.float32)
    batch_sounds = batch_sounds[:, :, :, :3]
    return batch_sounds
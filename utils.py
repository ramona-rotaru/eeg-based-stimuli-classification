import scipy.io
import numpy as np
import itertools
from EEGModels import EEGNet
from tensorflow.keras.utils import to_categorical
# import sklearn
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime
from numpy import load
from scipy.io import loadmat
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns


def get_tensorboard_callback():
    log_dir = "log/log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback


def shift_labels(labels):
    nb_classes = np.unique(labels)
    nb_classes = len(nb_classes)-1
    shifted_labels = np.array([int(np.ceil((nb_classes*((label-np.min(labels))))/(np.max(labels)-np.min(labels))))for label in labels])
    return shifted_labels


def load_data(annots, data):
    for value in annots[data]:
        data = value
    return data


def delete_blank_epochs(data):
    data1 = np.delete(data[0], np.s_[0:499], 1)
    data1 = np.delete(data1, np.s_[1500:2500], 1)
    return data1


def delete_blank_epochs2(data):
    data1 = np.delete(data, np.s_[0:499], 1)
    # print(data1.shape)
    data1 = np.delete(data1, np.s_[1500:2500], 1)
    # print(data1.shape)
    return data1


def swap_axes(data):
    data2 = np.swapaxes(data, 0, 1)
    # print(data2.shape)
    data2 = np.swapaxes(data2, 0, 2)
    # print(data2.shape)
    return data2


def extract_nans(data_name, labels):
    data = np.load(data_name)
    position = np.where(np.isnan(data[:, 0, 0]))[0]

    if len(position) == 0:
        return data, labels

    index_positions = position
    print(index_positions)
    contor = 0
    for index in index_positions:
        index = index-contor
        data = np.delete(data, index, axis=0)
        del labels[index]
        contor += 1
    return data, labels

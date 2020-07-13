import os
import time
import tqdm
import warnings ; warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
from random import shuffle
from distutils.version import LooseVersion


## Define class Dataset
class Dataset(object):
    def __init__(self, data, labels = None, width = 28, height = 28, max_value = 255, channels = 3):
        # Record image specification.
        self.IMAGE_WITH = width
        self.IMAGE_HEIGHT = height
        self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WITH, self.IMAGE_HEIGHT, self.CHANNELS
        if self.CHANNELS == 3:
            self.image_mode = 'RGB'
            self.cmap = None
        elif self.CHANNELS == 1:
            self.image_mode = 'L'
            self.cmap = 'gray'

        # If the image is different in size, resize it.
        if data.shape[1] != self.IMAGE_HEIGHT or data.shape[2] != self.IMAGE_WITH:
            data = self.image_resize(data, self.IMAGE_HEIGHT, self.IMAGE_WITH)
        # Store mixed data separately.
        index = list(range(len(data)))
        shuffle(index)
        self.data = data[index]

        if len(labels) > 0:
            # Save mixed labels separately.
            self.labels = labels[index]
            # List unique category values
            self.classes = np.unique(labels)
            # Create a one-hot encoding for each category based on the location in self.classes.
            one_hot = dict()
            no_classes = len(self.classes)
            for j, i in enumerate(self.classes):
                one_hot[i] = np.zeros(no_classes)
                one_hot[j] = 1.0
            self.one_hot = one_hot
        else:
            # Keep label variables as placeholders.
            self.labels = None
            self.classes = None
            self.one_hot = None
    
    def image_resize(self, dataset, newHeight, newWidth):
        # If necessary, resize image.
        channels = dataset.shape[3]
        images_resized = np.zeros([0, newHeight, newWidth, channels], dtype = np.uint8)
        for image in range(dataset.shape[0]):
            if channels == 1:
                temp = imresize(dataset[image][:, :, 0], [newHeight, newWidth], 'nearest')
                temp = np.expand_dims(temp, axis = 2)
            else:
                temp = imresize(dataset[image], [newHeight, newWidth], 'nearest')
            images_resized = np.append(images_resized, np.expand_dims(temp, axis = 0), axis = 0)
        return images_resized

    def get_batches(self, batch_size):
        # Importing batch of images and labels.
        current_index = 0
        # Verify that batch remains to be imported.
        while current_index < self.shape[0]:
            if current_index + batch_size > self.shape[0]:
                batch_size = self.shape[0] - current_index
            data_batch = self.data[current_index:current_index + batch_size]

            if len(self.labels) > 0:
                y_batch = np.array([self.one_hot[k] for k in \
                                    self.labels[current_index:current_index + batch_size]])
            else:
                y_batch = np.array([])
            current_index += batch_size
            yield (data_batch / self.IMAGE_MAX_VALUE) - 0.5, y_batch
import os
import gzip
import tarfile
import zipfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tqdm import tqdm
from CGan import CGan, Dataset
path = 'C:/Users/bitcamp/Desktop/서누/GAN/'
os.chdir(path)

class TqdmUpTo(tqdm):
    def update_to(self, b = 1, bsize = 1, tsize = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

## Load data
labels_filename = 'train-labels-idx1-ubyte.gz'
images_filename = 'train-images-idx3-ubyte.gz'

url = 'http://yann.lecun.com/exdb/mnist/'
with TqdmUpTo() as t:       # All Selective Keyword Factor
    urllib.request.urlretrieve(url + images_filename,
                               'MNIST_' + images_filename,
                               reporthook = t.update_to, data = None)

with TqdmUpTo() as t:       # All Selective keyword Kwargs
    urllib.request.urlretrieve(url + labels_filename,
                               'MNIST_' + labels_filename,
                               reporthook = t.update_to, data = None)

labels_path = './MNIST_train-labels-idx1-ubyte.gz'
images_path = './MNIST_train-images-idx3-ubyte.gz'

with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)

with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype = np.uint8,
                offset = 16).reshape(len(labels), 28, 28, 1)

batch_size = 32
z_dim = 96
epochs = 16

dataset = Dataset(images, labels, channels = 1)
gan = CGan(dataset, epochs, batch_size, z_dim, generator_name = 'mnist')

gan.show_original_images(25)
gan.fit(learning_rate = 0.0002, beta1 = 0.35)
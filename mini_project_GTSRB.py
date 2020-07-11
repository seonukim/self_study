## GTSRB ; (German Traffic Sign Recognition Benchmark)

# image preprocessing
N_CLASSES = 43
RESIZED_IMAGE = (32, 32)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple          ## for onehotencoding
path = '/Users/seonwoo/Downloads/GTSRB/Final_Training/'
os.chdir(path)
np.random.seed(101)

Dataset = namedtuple('Dataset', ['X', 'y'])

def to_tf_format(imgs):
    return np.stack([img[:, :, np.newaxis] for img in imgs], axis = 0).astype(np.float32)

def read_dataset_ppm(rootpath, n_labels, resize_to):
    images = []
    labels = []
    path = '/Users/seonwoo/Downloads/GTSRB/Final_Training/Images'

    for c in range(n_labels):
        full_path = path + format(c, '05d') + '/'
        for img_name in glob.glob(full_path + '*.ppm'):
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img / 255.0)[:, :, 0]

            if resize_to:
                img = resize(img, resize_to, mode = 'reflect')

            label = np.zeros((n_labels, ), dtype = np.float32)
            label[c] = 1.0

            images.append(img.astype(np.float32))
            labels.append(label)
    
    return Dataset(X = to_tf_format(images).astype(np.float32),
                   y = np.matrix(labels).astype(np.float32))

dataset = read_dataset_ppm('GTSRB/Final_Training/Images', N_CLASSES, RESIZED_IMAGE)
print(dataset.X.shape)                  # (39209, 32, 32, 1)
print(dataset.y.shape)                  # (39209, 43)

## Printing the first image
plt.imshow(dataset.X[0, :, :, :].reshape(RESIZED_IMAGE))    # first image
plt.show()
print(dataset.y[0, :])      # label

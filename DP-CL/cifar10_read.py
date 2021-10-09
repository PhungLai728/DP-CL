########################################################################
#
# Functions for downloading the CIFAR-10 data-set from the internet
# and loading it into memory.
#
# Implemented in Python 3.5
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set
#    if it is not already located in the given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels
#    for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for the training- and test-sets are returned as 4-dim numpy
# arrays each with the shape: [image_number, height, width, channel]
# where the individual pixels are floats between 0.0 and 1.0.
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import numpy as np
import pickle
import os
#import download
from six.moves.urllib.request import urlretrieve
import random
# from dataset import one_hot_encoded
import sklearn
from sklearn.preprocessing import OneHotEncoder
import cv2
import data_utils
import tarfile
import zipfile
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

#from PIL import ImageEnhance, Image, ImageOps

########################################################################

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
# data_path = "Cifar/cifar10/" #"data/CIFAR-10/"
data_path = "CIFAR_data/cifar_10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_10_DIR = "/cifar_10"
CIFAR_100_DIR = "/cifar_100"
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
########################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32
crop_imageSize = 28

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Number of Training examples
_image_for_training = 30000
_image_for_testing = 5000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
# Private functions for downloading, unpacking and loading data-files.

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

def _get_file_path(filename=""):
    """
        Return the full path of a data-file for the data-set.
        
        If filename=="" then return the directory of the files.
        """
    
    #return os.path.join("/cifar-10-batches-bin", filename)
    return os.path.join(data_path, filename)


def _unpickle(filename):
    """
        Unpickle the given file and return the data.
        
        Note that the appropriate dir-name is prepended the filename.
        """
    
    # Create full path for the file.
    file_path = _get_file_path(filename)
    print(filename)
    print("Loading data: " + file_path)
    
    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        data = pickle.load(file, encoding='bytes')
        # data = pickle.load(file)
    
    return data

def randomCrop(img, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width, ]
    return img

def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image

def augment_brightness_camera_images(image):
    #image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image[:,:,2] = image[:,:,2]*random_bright
    #image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    image = np.clip(image, -1.0, 1.0)
    return image

'''def RandomBrightness(image, min_factor, max_factor):
    """
    Random change the passed image brightness.
    :param images: The image to convert into monochrome.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
    PIL.Image.
    """
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_brightness = ImageEnhance.Brightness(image)
    return image_enhancer_brightness.enhance(factor)

def RandomContrast(image, min_factor, max_factor):
    """
    Random change the passed image contrast.
    :param images: The image to convert into monochrome.
    :type images: List containing PIL.Image object(s).
    :return: The transformed image(s) as a list of object(s) of type
    PIL.Image.
    """
    factor = np.random.uniform(min_factor, max_factor)

    image_enhancer_contrast = ImageEnhance.Contrast(image)
    return image_enhancer_contrast.enhance(factor)
'''

def _convert_images(raw, train_test):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 127.5 - 1.0
    
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    
    # Random crop the images
    _images_ = []
    if train_test == True:
        for i in range(images.shape[0]):
            #image = horizontal_flip(images[i], rate=0.5)
            #image = augment_brightness_camera_images(image)
            #image = RandomContrast(image, 0.8, 1.8)
            #_images_.append(randomCrop(image, crop_imageSize, crop_imageSize))
            _images_.append(randomCrop(images[i], crop_imageSize, crop_imageSize))
        _images_ = np.asarray(_images_)
    else:
        for i in range(images.shape[0]):
            _images_.append(randomCrop(images[i], crop_imageSize, crop_imageSize))
        _images_ = np.asarray(_images_)
    del images
    _images_ -= np.mean(_images_)
    _images_ /= np.std(_images_, axis = 0)

    p_images = _images_.reshape([-1, num_channels * crop_imageSize * crop_imageSize])
    return p_images


def _load_data(filename, train_test):
    """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """
    
    # Load the pickled data-file.
    data = _unpickle(filename)
    
    # Get the raw images.
    raw_images = data[b'data']
    
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels'])
    
    # Convert the images.
    images = _convert_images(raw_images, train_test)
    
    return images, cls


########################################################################
# Public functions that you may call to download the data-set from
# the internet and load the data into memory.


'''def maybe_download_and_extract():
    """
        Download and extract the CIFAR-10 data-set if it doesn't already exist
        in data_path (set this variable first to the desired path).
        """
    
    download.maybe_download_and_extract(url=data_url, download_dir=data_path)
    '''

def _print_download_progress(count, block_size, total_size):
    """
        Show the download progress of the cifar data
        """
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()

def _cifar_maybe_download_and_extract(data_dir):
    """
        Routine to download and extract the cifar dataset
        
        Args:
        data_dir      Directory where the downloaded data will be stored
        """
    cifar_10_directory = data_dir + CIFAR_10_DIR
    cifar_100_directory = data_dir + CIFAR_100_DIR
    
    # If the data_dir does not exist, create the directory and download
    # the data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
        url = CIFAR_10_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
        
        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")
        
        url = CIFAR_100_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)
        
        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        os.rename(data_dir + "/cifar-10-batches-py", cifar_10_directory)
        print(cifar_10_directory)
        os.rename(data_dir + "/cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)
    return cifar_10_directory

def load_class_names():
    """
        Load the names for the classes in the CIFAR-10 data-set.
        
        Returns a list with the names. Example: names[3] is the name
        associated with class-number 3.
        """
    
    # Load the class-names from the pickled file.
    raw = _unpickle(filename="batches.meta")[b'label_names']
    # raw = _unpickle(filename="batches.meta")['label_names']
    
    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]
    
    return names


def load_training_data():
    """
        Load all the training-data for the CIFAR-10 data-set.
        
        The data-set is split into 5 data-files which are merged here.
        
        Returns the images, class-numbers and one-hot encoded class-labels.
        """
    
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, crop_imageSize * crop_imageSize * num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)
    
    # Begin-index for the current batch.
    begin = 0
    
    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1), train_test = True)
        
        # Number of images in this batch.
        num_images = len(images_batch)
        
        # End-index for the current batch.
        end = begin + num_images
        
        # Store the images into the array.
        images[begin:end, :] = images_batch
        
        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch
        
        # The begin-index for the next batch is the current end-index.
        begin = end
    
    _cls = one_hot_encoded(class_numbers=cls, num_classes=num_classes)
    perm_inds = np.arange(images.shape[0])
    # print('perm_inds',perm_inds)
    # exit()
    np.random.shuffle(perm_inds)
    images = images[perm_inds]
    _cls = _cls[perm_inds]
    _images = images[:_image_for_training,:]
    _cls = _cls[:_image_for_training,:]
    
    return _images, cls, _cls
    #return _images, cls, one_hot_encoded(class_numbers=_cls, num_classes=num_classes)


def load_test_data():
    """
        Load all the test-data for the CIFAR-10 data-set.
        
        Returns the images, class-numbers and one-hot encoded class-labels.
        """
    
    images, cls = _load_data(filename="test_batch", train_test = False)
    
    _cls = one_hot_encoded(class_numbers=cls, num_classes=num_classes)
    perm_inds = np.arange(images.shape[0])
    np.random.shuffle(perm_inds)
    images = images[perm_inds]
    _cls = _cls[perm_inds]
    _images = images[:_image_for_testing,:]
    _cls = _cls[:_image_for_testing,:]
    
    return _images, cls, _cls

class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            #assert images.shape[3] == 3
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        """if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]"""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            """# Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]"""
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    #data_path = _cifar_maybe_download_and_extract('Cifar') #
    #maybe_download_and_extract()
    load_class_names()

    train_images, _, train_labels = load_training_data()
    test_images, _, test_labels = load_test_data()
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
########################################################################






























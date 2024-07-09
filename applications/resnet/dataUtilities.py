## Standard libraries
import os
import numpy as np
from PIL import Image

## JAX
import jax
import jax.numpy as jnp
from jax import random

## PyTorch
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

def numpy_collate(batch):
	if isinstance(batch[0], np.ndarray):
		return np.stack(batch)
	elif isinstance(batch[0], (tuple,list)):
		transposed = zip(*batch)
		return [numpy_collate(samples) for samples in transposed]
	else:
		return np.array(batch)


class NumpyLoader(data.DataLoader):
	def __init__(self, dataset, batch_size=1,
				shuffle=False, sampler=None,
				batch_sampler=None, num_workers=0,
				pin_memory=False, drop_last=False,
				timeout=0, worker_init_fn=None):
		super(self.__class__, self).__init__(dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			sampler=sampler,
			batch_sampler=batch_sampler,
			num_workers=num_workers,
			collate_fn=numpy_collate,
			pin_memory=pin_memory,
			drop_last=drop_last,
			timeout=timeout,
			worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
	def __call__(self, pic):
		return np.ravel(np.array(pic, dtype=jnp.float32))

class JustCast(object):
	def __call__(self, pic):
		return np.expand_dims(np.array(pic, dtype=jnp.float32),axis = -1)

def get_cifar10_dataset(path_prefix = ''):
	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
	DATASET_PATH = path_prefix + "data1/"
	################################################################################

	DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
	DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])


	# Transformations applied on each image => bring them into a numpy array
	def image_to_numpy(img):
		img = np.array(img, dtype=np.float32)
		img = (img / 255. - DATA_MEANS) / DATA_STD

		return img

	train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])
	
	test_transform = image_to_numpy

	train_set = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
	# Loading the test set
	test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
	return train_set,  test_set

def get_cifar10_dataset_unmodified(path_prefix = '/workspace/tom/resnet_stuff/',whiten = False):
	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
	DATASET_PATH = path_prefix + "data/"
	################################################################################
	if whiten:
		train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
		DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
		DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
		print("Data mean", DATA_MEANS)
		print("Data std", DATA_STD)
	else:
		DATA_MEANS = 0.0
		DATA_STD = 1.0

	# DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
	# DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])

	# Transformations applied on each image => bring them into a numpy array
	def image_to_numpy(img):
		img = np.array(img, dtype=np.float32)
		if whiten:
			img = (img / 255. - DATA_MEANS) / DATA_STD
		else:
			img = img / 255.
		return img

	train_transform = image_to_numpy


	test_transform = image_to_numpy
	# Loading the training dataset. We need to split it into a training and validation part
	# We need to do a little trick because the validation set should not use the augmentation.
	train_set = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
	# Loading the test set
	test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
	return train_set,  test_set


def get_cifar100_dataset(path_prefix = ''):
	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
	DATASET_PATH = path_prefix + "data1/"
	################################################################################

	DATA_MEANS = np.array([0.5071, 0.4867, 0.4408])
	DATA_STD = np.array([0.2675, 0.2565, 0.2761])


	# Transformations applied on each image => bring them into a numpy array
	def image_to_numpy(img):
		img = np.array(img, dtype=np.float32)
		img = (img / 255. - DATA_MEANS) / DATA_STD

		return img

	train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      image_to_numpy
                                     ])
	
	test_transform = image_to_numpy

	train_set = CIFAR100(root=DATASET_PATH, train=True, transform=train_transform, download=True)
	# Loading the test set
	test_set = CIFAR100(root=DATASET_PATH, train=False, transform=test_transform, download=True)
	return train_set,  test_set


def get_cifar100_dataset_unmodified(path_prefix = '/workspace/tom/resnet_stuff/',whiten = False):
	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
	DATASET_PATH = path_prefix + "data/"
	################################################################################
	if whiten:
		train_dataset = CIFAR100(root=DATASET_PATH, train=True, download=True)
		DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
		DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
		print("Data mean", DATA_MEANS)
		print("Data std", DATA_STD)
	else:
		DATA_MEANS = 0.0
		DATA_STD = 1.0

	# DATA_MEANS = np.array([0.5071, 0.4867, 0.4408])
	# DATA_STD = np.array([0.2675, 0.2565, 0.2761])
	
	# Transformations applied on each image => bring them into a numpy array
	def image_to_numpy(img):
		img = np.array(img, dtype=np.float32)
		if whiten:
			img = (img / 255. - DATA_MEANS) / DATA_STD
		return img
	train_transform = image_to_numpy
	test_transform = image_to_numpy
	# Loading the training dataset. We need to split it into a training and validation part
	# We need to do a little trick because the validation set should not use the augmentation.
	train_set = CIFAR100(root=DATASET_PATH, train=True, transform=train_transform, download=True)
	# Loading the test set
	test_set = CIFAR100(root=DATASET_PATH, train=False, transform=test_transform, download=True)
	return train_set,  test_set


def get_mnist_dataset(mnist_data_dir):


	try:
		mnist_dataset = MNIST(mnist_data_dir, download=False,transform=JustCast())
	except:
		mnist_dataset = MNIST(mnist_data_dir, download=True,transform = JustCast())

	# Get full test dataset
	try:
		mnist_dataset_test = MNIST(mnist_data_dir, download=False, train=False,transform = JustCast())
	except:
		mnist_dataset_test = MNIST(mnist_data_dir, download=True, train=False,transform = JustCast())


	return mnist_dataset, mnist_dataset_test


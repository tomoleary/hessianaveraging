# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import sys, os
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-method', '--method', type=str, default='gd', help="What method")
parser.add_argument('-batch_size', '--batch_size', type = int, default = 32, help = 'gradient batch size')
parser.add_argument('-hbatch_size', '--hbatch_size', type = int, default = 32, help = 'Hessian batch size')
parser.add_argument('-num_epochs', '--num_epochs', type=int, default=50, help="How many epochs")
parser.add_argument('-step_size', '--step_size', type=float, default=1e-2, help="What step size or 'learning rate'?")
parser.add_argument('-k_rank', '--k_rank', type=int, default=1, help="Rank for Hessian approximation")
parser.add_argument('-gamma_damping', '--gamma_damping', type=float, default=1e-4, help="L-M damping")
parser.add_argument('-epsilon', '--epsilon', type=float, default=1e-8, help="What epsilon for adaptive methods")
parser.add_argument('-hess_frequency', '--hess_frequency', type=int, default=1, help="Hessian update frequency")
parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")
parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")
parser.add_argument('-regularization', '--regularization', type=float, default=0.0, help="Tikhonov regularization parameter")
parser.add_argument('-dataset_dir', '--dataset_dir', type=str, default='/storage/tom/resnet_stuff/', help="Where to store the data")

parser.add_argument('-lr_schedule', '--lr_schedule', type=str, default='piecewise', help="Use LR Schedule or do not")
parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.0, help="Weight decay parameter")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.visible_gpu)

import time

import math
from functools import partial
from collections import defaultdict
from typing import Any, Sequence
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from flax import linen as nn
from flax.training import train_state, common_utils
import optax

import pickle
jax.local_devices()

################################################################################
# Load custom optimizers
sys.path.append('../../')
sys.path.append(os.environ.get('HESSAVG_PATH'))
from hessianaveraging import Optimizer, Adam, GradientDescent, AdaGrad, RMSProp, AdaHessian, DiagonalNewton,\
					 DiagonallyAveragedNewton

################################################################################
# Run hyperparameters
method = args.method
assert method in ['adam', 'adahessian','adagrad','gd','mgd','dnewton','dan','dan2','lrsfn','rmsprop']
num_epochs = args.num_epochs
step_size = args.step_size
k_rank = args.k_rank
gamma_damping = args.gamma_damping
epsilon = args.epsilon
num_epochs = args.num_epochs
batch_size = args.batch_size
hbatch_size = args.hbatch_size
hessian_frequency = args.hess_frequency


if method in ['dan','dan2','dnewton','adahessian']:
		assert hbatch_size <= batch_size


################################################################################
# Get the data
sys.path.append('../')

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.utils.data as data

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

mnist_data_dir = 'data/mnist/'
os.makedirs(mnist_data_dir,exist_ok=True)

# mnist_dataset, mnist_dataset_test = get_mnist_dataset(mnist_data_dir)

mnist_dataset = MNIST(mnist_data_dir, download=True,transform=JustCast())
mnist_dataset_test = MNIST(mnist_data_dir, download = True,train=False,transform = JustCast())
	
	
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

testing_generator = NumpyLoader(mnist_dataset_test, batch_size=batch_size, num_workers=0)

train_images = np.expand_dims(np.array(mnist_dataset.train_data),axis = -1)
train_labels = np.array(mnist_dataset.train_labels)



test_images = np.expand_dims(np.array(mnist_dataset_test.test_data),axis = -1)
test_labels = np.array(mnist_dataset_test.test_labels)


# test_generator = NumpyLoader(mnist_dataset_test, batch_size=batch_size, num_workers=0)

all_train = {'image':train_images,'label':train_labels}
all_test = {'image':test_images, 'label':test_labels}

image_size = 28
n_outputs = 10
n_channels = 1 

################################################################################

use_batchnorm = True

from flax import linen as nn  # Linen API

class CNN(nn.Module):
	"""A simple CNN model."""
	use_batchnorm: False

	@nn.compact
	def __call__(self, x,train):
		x = nn.Conv(features=32, kernel_size=(3, 3))(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = nn.Conv(features=64, kernel_size=(3, 3))(x)
		if self.use_batchnorm:
			x = nn.Dropout(rate=0.5, deterministic=not train)(x)
			x = nn.BatchNorm(use_running_average=not train)(x)
		x = nn.relu(x)
		x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = x.reshape((x.shape[0], -1))  # flatten
		x = nn.Dense(features=256)(x)
		x = nn.relu(x)
		x = nn.Dense(features=10)(x)
		return x

# instance the network 
model = CNN(use_batchnorm)
if False:
	print(model.tabulate(jax.random.PRNGKey(0), jnp.ones((1, image_size, image_size, 1))))

################################################################################
# Initialize weights and define training procedure


init_rng = jax.random.PRNGKey(0)
variables = model.init(init_rng, jnp.ones([1, image_size, image_size, n_channels]), train = False)
del init_rng

params = variables['params']

################################################################################
# Loss function


def loss_fn(network, params, batch, batch_stats = None,train = True):
	assert batch_stats is not None
	logits, batch_stats = network.apply({'params': params, 'batch_stats': batch_stats}, batch['image'],\
			rngs={'dropout':jax.random.PRNGKey(0)}, mutable = ['batch_stats'],train = train)
	loss = optax.softmax_cross_entropy_with_integer_labels(
		logits=logits, labels=batch['label']).mean()
	_, batch_stats = batch_stats.pop('batch_stats')
	return loss, batch_stats

################################################################################
# Train state only used to make metrics faster

tx = optax.adam(learning_rate=args.step_size)

class TrainState(train_state.TrainState):
		batch_stats: Any

state = TrainState.create(
		apply_fn = model.apply,
		params = variables['params'],
		batch_stats = variables['batch_stats'],
		tx = tx)

def compute_loss_acc(logits, labels):
	one_hot = jax.nn.one_hot(labels, n_outputs)
	loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
	accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
	return loss, accuracy


@jax.jit
def eval_step(state, batch, params, batch_stats):
	variables = {'params': params, 'batch_stats': batch_stats}
	logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
	metrics = compute_loss_acc(logits, batch['label'])
	return metrics

def compute_metrics(network, params, data_iterator, batch_stats = None):
	assert batch_stats is not None
	losses = []
	accs = []

	for step, (x,y) in enumerate(data_iterator):
			data = {'image':x, 'label':y}
			loss,acc = eval_step(state,data,params,batch_stats)
			losses.append(loss)
			accs.append(acc)
	return {'loss':np.mean(losses), 'acc':np.mean(accs)}, batch_stats


################################################################################
# Begin instancing optimizer, batch_stats, etc.

if use_batchnorm:
    batch_stats = variables['batch_stats']
    loss = lambda params, batch, batch_stats: loss_fn(model, params, batch, batch_stats)
else:
    # Loss function
    loss = lambda params, batch: loss_fn(model, params, batch)

# LR schedule
train_steps_per_epoch = len(training_generator)
num_train_steps = train_steps_per_epoch * num_epochs

# LR scheduling

if args.lr_schedule == 'piecewise':
	print('Using piecewise lr schedule')
	lr_schedule = optax.piecewise_constant_schedule(init_value=step_size,
						 boundaries_and_scales={int(num_train_steps*0.25):0.25,
												int(num_train_steps*0.5):0.25,
												int(num_train_steps*0.75):0.25})
elif args.lr_schedule == 'cosine':
	print('Using cosine lr schedule')
	lr_schedule = optax.cosine_onecycle_schedule(transition_steps=num_train_steps, peak_value=step_size)
else:
	print('Using no lr schedule')
	lr_schedule = None
		
# Weight decay
if args.weight_decay > 0.0:
		weight_decay = args.weight_decay
else:
		weight_decay = None


if method.lower() == 'adam':
	optimizer = Adam(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
											weight_decay = weight_decay)
elif method.lower() == 'adagrad':
	optimizer = AdaGrad(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
										weight_decay = weight_decay)
elif method.lower() == 'adahessian':
	optimizer = AdaHessian(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank,epsilon = epsilon,\
															 hessian_frequency = 1)
elif method.lower() == 'dnewton':
	optimizer = DiagonalNewton(loss, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank,gamma_damping = gamma_damping)
elif method.lower() == 'dan':
	optimizer = DiagonallyAveragedNewton(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
							 k_rank = k_rank, gamma_damping = gamma_damping,hessian_frequency = 1)
elif method.lower() == 'dan2':
	optimizer = DiagonallyAveragedNewton(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank, gamma_damping = gamma_damping,\
															 norm_exponent = 2,hessian_frequency = 1)
elif method.lower() == 'gd':
	optimizer = GradientDescent(loss,step_size = step_size)
elif method.lower() == 'mgd':
	optimizer = MomentumGradientDescent(loss,params, lr_schedule = lr_schedule, step_size = step_size)
		
elif method.lower() == 'rmsprop':
	optimizer = RMSProp(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
										weight_decay = weight_decay)
else:
	raise

# Prints
print(80*'#')
preamble = 'Successfully built '+method.lower()+' object'
print(preamble.center(80))
lr_print = 'learning rate = '+str(step_size)
print(lr_print.center(80))
if method.lower() in ['adahessian','dnewton','dan']:
	k_rank_print = 'k_rank = '+str(k_rank)
	print(k_rank_print.center(80))


################################################################################
# Setup for training
metrics_history = {'train_loss': [],
									 'train_accuracy': [],
									 'test_loss': [],
									 'test_accuracy': [],
									 'epoch_time': [],
									 'epoch': []}

# Initial evaluations
metric_t0 = time.time()
metrics_train, batch_stats = compute_metrics(model,params,training_generator,batch_stats = batch_stats)
metrics_test, batch_stats = compute_metrics(model,params,testing_generator, batch_stats = batch_stats)
metric_time = time.time() - metric_t0
metrics_history['train_loss'].append(np.array(metrics_train['loss']))
metrics_history['train_accuracy'].append(np.array(metrics_train['acc']))
metrics_history['test_loss'].append(np.array(metrics_test['loss']))
metrics_history['test_accuracy'].append(np.array(metrics_test['acc']))
print(f"train epoch: {0}, "
			f"loss: {metrics_history['train_loss'][-1]}, "
			f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
print(f"test epoch: {0}, "
			f"loss: {metrics_history['test_loss'][-1]}, "
			f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
print('Max test accuracy = ',100*np.max(np.array(metrics_history['test_accuracy'])))
print('The metrics took', metric_time, 's')

# Training loop
for epoch in range(1,num_epochs):
	if method.lower() in ['adahessian','dan', 'dan2']:
		if epoch > 1:
			optimizer.hessian_frequency = hessian_frequency
	start_time = time.time()
	for step,(x,y) in enumerate(training_generator):
		batch = {'image':x, 'label':y}
		if hbatch_size < batch_size:
			hbatch = {'image':x[:hbatch_size],'label':y[:hbatch_size]}
		else:
			hbatch = batch
		params, batch_stats = optimizer.update(params,batch,batch_stats= batch_stats)
		# _, batch_stats = batch_stats.pop('batch_stats')

	epoch_time = time.time() - start_time
	print('The last epoch took', epoch_time, 's')

	metric_t0 = time.time()
	# Post-process and compute metrics after each epoch
	metrics_train, batch_stats = compute_metrics(model,params,training_generator,batch_stats = batch_stats)
	metrics_test, batch_stats = compute_metrics(model,params,testing_generator, batch_stats = batch_stats)
	metric_time = time.time() - metric_t0

	metrics_history['train_loss'].append(np.array(metrics_train['loss']))
	metrics_history['train_accuracy'].append(np.array(metrics_train['acc']))
	metrics_history['test_loss'].append(np.array(metrics_test['loss']))
	metrics_history['test_accuracy'].append(np.array(metrics_test['acc']))
	metrics_history['epoch_time'].append(epoch_time)
	metrics_history['epoch'].append(epoch)


	print(f"train epoch: {epoch}, "
				f"loss: {metrics_history['train_loss'][-1]}, "
				f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
	print(f"test epoch: {epoch}, "
				f"loss: {metrics_history['test_loss'][-1]}, "
				f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
	print('Max test accuracy = ',100*np.max(np.array(metrics_history['test_accuracy'])))
	print('The metrics took', metric_time, 's')



################################################################################
# Post-processing
logger = metrics_history
run_name = method+'_lr'+str(step_size)+'_n'+str(num_epochs)+'_ds'+str(args.run_seed)

if method in ['adahessian','dnewton','dan', 'dan2']:
		run_name += '_k'+str(k_rank)+'_g'+str(gamma_damping)  

if method in ['adahessian','dan','dan2'] and args.hess_frequency > 1:
	run_name += '_hf'+str(args.hess_frequency)

logging_dir = 'mnist_classification_logging/'
os.makedirs(logging_dir,exist_ok = True)
logger_name = run_name+'.pkl'

with open(logging_dir+logger_name, 'wb+') as f:
		pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)

print(80*'#')
print('Run finished and saved successfully'.center(80))



try:
	import matplotlib.pyplot as plt  # Visualization

	# Plot loss and accuracy in subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
	ax1.set_title('Loss')
	ax2.set_title('Accuracy')
	for dataset in ('train','test'):
		ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
		ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
	ax1.legend()
	ax2.legend()
	plt.show()
	plt.clf()

except:
	print('Matplotlib plotting issue'.center(80))






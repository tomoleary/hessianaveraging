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
parser.add_argument('-num_epochs', '--num_epochs', type=int, default=10, help="How many epochs")
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
from hessavg import Optimizer, Adam, GradientDescent, AdaGrad, RMSProp, AdaHessian, DiagonalNewton,\
					 DiagonallyAveragedNewton

################################################################################
# Run hyperparameters
method = args.method
assert method in ['adam', 'adahessian','adagrad','gd','mgd','dnewton','dan','dan2','rmsprop']
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
t0 = time.time()

IMAGE_SIZE = 32
NUM_CLASSES = 10

n_outputs = NUM_CLASSES

from dataUtilities import *

train_set, test_set = get_cifar10_dataset(path_prefix = args.dataset_dir)

training_generator = data.DataLoader(train_set,
									 batch_size=batch_size,
									 shuffle=True,
									 drop_last=True,
									 collate_fn=numpy_collate,
									 num_workers=1,
									 persistent_workers=True,
									 worker_init_fn = np.random.seed(args.run_seed))


htraining_generator = data.DataLoader(train_set,
									 batch_size=hbatch_size,
									 shuffle=True,
									 drop_last=True,
									 collate_fn=numpy_collate,
									 num_workers=1,
									 persistent_workers=True,
									 worker_init_fn = np.random.seed(args.run_seed))

testing_generator = data.DataLoader(test_set,
									 batch_size=batch_size,
									 shuffle=True,
									 drop_last=True,
									 collate_fn=numpy_collate,
									 num_workers=1,
									 persistent_workers=True,
									 worker_init_fn = np.random.seed(0))

################################################################################

from ResNet import ResNet
# Taken from here: https://juliusruseckas.github.io/ml/flax-cifar10.html

model = ResNet(NUM_CLASSES,
							 channel_list = [64, 128, 256, 512],
							 num_blocks_list = [2, 2, 2, 2],
							 strides = [1, 1, 2, 2, 2],
							 head_p_drop = 0.3)

rng = jax.random.PRNGKey(args.run_seed)

params_rng, dropout_rng = jax.random.split(rng)

@jax.jit
def initialize(params_rng):
		init_rngs = {'params': params_rng}
		input_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 3)
		variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32), train=False)
		return variables

################################################################################
# Initialize weights and define training procedure

variables = initialize(params_rng)
params = variables['params']

################################################################################
# Loss function


def loss_fn(network, params, batch, batch_stats = None,train = True):
		assert batch_stats is not None
		logits, batch_stats = network.apply({'params': params, 'batch_stats': batch_stats}, batch['image'],\
						rngs={'dropout':jax.random.PRNGKey(0)}, mutable = ['batch_stats'],train = train)
		one_hot = jax.nn.one_hot(batch['label'], n_outputs)
		loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
		# Adding regularization
		if args.regularization > 0.0:
				rav_param, unravel = ravel_pytree(params)
				loss += args.regularization*(rav_param**2).sum()
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

batch_stats = variables['batch_stats']
loss = lambda params, batch, batch_stats: loss_fn(model, params, batch, batch_stats)

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
															 k_rank = k_rank,epsilon = epsilon)
elif method.lower() == 'dnewton':
		optimizer = DiagonalNewton(loss, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank,gamma_damping = gamma_damping)
elif method.lower() == 'dan':
		optimizer = DiagonallyAveragedNewton(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank, gamma_damping = gamma_damping)
elif method.lower() == 'dan2':
		optimizer = DiagonallyAveragedNewton(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
															 k_rank = k_rank, gamma_damping = gamma_damping,\
															 norm_exponent = 2)
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
preamble = 'Successfully built '+method.upper()+' object'
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
	for step, (gdata, hdata) in enumerate(zip(training_generator,htraining_generator)):
			batch = {'image':gdata[0], 'label':gdata[1]}
			hbatch = {'image':hdata[0], 'label': hdata[1]}

			params, batch_stats = optimizer.update(params,batch,hess_batch = hbatch, batch_stats= batch_stats)
			_, batch_stats = batch_stats.pop('batch_stats')

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

run_name += 'g'+str(batch_size)+'h'+str(hbatch_size)

logging_dir = 'cifar10_resnet_logging/'
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






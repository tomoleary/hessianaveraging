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
parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")
parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")
parser.add_argument('-regularization', '--regularization', type=float, default=0.0, help="Tikhonov regularization parameter")
parser.add_argument('-dataset_dir', '--dataset_dir', type=str, default='/storage/tom/resnet_stuff/', help="Where to store the data")

parser.add_argument('-lr_schedule', '--lr_schedule', type=str, default='None', help="Use LR Schedule or do not")
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
sys.path.append('../../../')
sys.path.append(os.environ.get('HESSAVG_PATH'))
from hessavg import Optimizer, Adam, GradientDescent, AdaGrad, RMSProp, AdaHessian, DiagonalNewton,\
					 DiagonallyAveragedNewton, FullNewton, FullyAveragedNewton


################################################################################
# Run hyperparameters
method = args.method
assert method in ['adam', 'adahessian','adagrad','gd','mgd','dnewton','dan','dan2','rmsprop','newton','fan']
num_epochs = args.num_epochs
step_size = args.step_size
k_rank = args.k_rank
gamma_damping = args.gamma_damping
epsilon = args.epsilon
num_epochs = args.num_epochs
batch_size = args.batch_size
hbatch_size = args.hbatch_size


if method in ['dan','dan2','dnewton','adahessian', 'newton','fan']:
		assert hbatch_size <= batch_size


################################################################################
# Get the data

ijcnn_data = np.load('ijcnn1.npz')
train_images = ijcnn_data['X']
train_labels = ijcnn_data['y']

n_data = train_images.shape[0]

train_images = jnp.array(train_images)
train_labels = jnp.array(train_labels)

ijcnn_data_test = np.load('ijcnn1_test.npz')
# print(list(ijcnn_data_test.keys()))
test_images = ijcnn_data_test['Xtest']
test_labels = ijcnn_data_test['ytest']

test_images = jnp.array(test_images)
test_labels = jnp.array(test_labels)

training_data = {'image':train_images,'label':train_labels}
testing_data = {'image':test_images,'label':test_labels}

input_size = 22

################################################################################

class LogisticRegressionModel(nn.Module):
	def setup(self):
		self.layer = nn.Dense(1,use_bias = False)

	@nn.compact
	def __call__(self, x):
		return jnp.squeeze(self.layer(x))

model = LogisticRegressionModel()

################################################################################
# Initialize weights and define training procedure

params = model.init(jax.random.PRNGKey(0), np.zeros((1,input_size)))
# print(params)
print(model.apply(params, np.zeros((1,input_size))))


################################################################################
# Loss function


def loss_fn(network, params, batch):
	label = batch['label']
	logits = network.apply(params, batch['image'])
	temp = jnp.multiply(label,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean() + (1./(2*len(batch['image'])))*jnp.linalg.norm(rav_param)**2
	return loss

################################################################################
# Train state only used to make metrics faster

tx = optax.adam(learning_rate=args.step_size)

class TrainState(train_state.TrainState):
	pass

state = TrainState.create(
		apply_fn = model.apply,
		params = params,
		tx = tx)

def compute_loss_acc(params,logits, labels):
	temp = jnp.multiply(labels,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean() + (1./2*len(labels))*jnp.linalg.norm(rav_param)**2

	positive = jnp.array(logits)>0.0
	preds = jnp.array(positive, dtype = int)
	label_bool = labels == 1
	acc = jnp.mean(label_bool == preds).mean()
	return loss, acc, losses.mean()



def compute_batched_errs(state, params, batch):
	logits = state.apply_fn(params, batch['image'])
	labels = batch['label']

	temp = jnp.multiply(labels,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean()  

	reg = (1./len(labels))*jnp.linalg.norm(rav_param)**2

	positive = jnp.array(logits)>0.0
	preds = jnp.array(positive, dtype = int)
	label_bool = labels == 1
	accs = (label_bool == preds)


	return losses + reg, accs, losses


@jax.jit
def eval_step(state, batch, params):
	logits = state.apply_fn(params, batch['image'])
	metrics = compute_loss_acc(params,logits, batch['label'])
	return metrics

@jax.jit
def compute_metrics(state, params, data, batch_size = 1024):
	losses = []
	accs = []
	losses_wo_reg = []

	losses = None
	accs = None
	losses_wo_reg = None
	
	n_data = data['image'].shape[0]
	n_batches, remainder = divmod(n_data, batch_size)


	for i_batch in range(n_batches):
		start = i_batch * batch_size 
		end = start + batch_size
		batch = jax.tree_map(lambda x : x[start:end], data) # no shuffling needed here
		# print('batch = ',batch)
		lossi,acci,lwri = compute_batched_errs(state,params,batch)
		if losses is None:
			losses = jnp.copy(lossi)
			accs = jnp.copy(acci)
			losses_wo_reg = jnp.copy(lwri)
		else:
			losses = jnp.concatenate([losses,lossi])
			accs = jnp.concatenate([accs,acci])
			losses_wo_reg = jnp.concatenate([losses_wo_reg,lwri])


	return {'loss':jnp.mean(losses), 'acc':jnp.mean(accs),'loss_without_reg':jnp.mean(losses_wo_reg)}


################################################################################
# Begin instancing optimizer etc.

loss = lambda params, batch: loss_fn(model, params, batch)

# LR schedule
train_steps_per_epoch = n_data
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
elif method.lower() == 'newton':
	optimizer = FullNewton(loss, lr_schedule = lr_schedule, step_size = step_size,\
															 gamma_damping = gamma_damping)
elif method.lower() == 'fan':
	optimizer = FullyAveragedNewton(loss, params, lr_schedule = lr_schedule, step_size = step_size,\
															gamma_damping = gamma_damping)
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
					 'epoch': [],
					 'train_loss_woreg':[],
					 'test_loss_woreg':[]}

# Initial evaluations
metric_t0 = time.time()
metrics_train = compute_metrics(state,params,training_data)
metric_time = time.time() - metric_t0
metrics_history['train_loss'].append(np.array(metrics_train['loss']))
metrics_history['train_accuracy'].append(np.array(metrics_train['acc']))
metrics_history['train_loss_woreg'].append(np.array(metrics_train['loss_without_reg']))

print(f"train epoch: {0}, "
			f"loss: {metrics_history['train_loss'][-1]}, "
			f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
print('The metrics took', metric_time, 's')

################################################################################
# First Training loop

rng_key = jax.random.PRNGKey(0)
hrng_key = jax.random.PRNGKey(1) # a different key for the hessian

n_batches, remainder = divmod(n_data, batch_size)

for epoch in range(1,num_epochs):

	rng_key, subkey = jax.random.split(rng_key)
	shuffled_inds = jax.random.permutation(subkey, n_data, axis=0)
	# Hessian data shuffling
	hrng_key, hsubkey = jax.random.split(hrng_key)
	hshuffled_inds = jax.random.permutation(hsubkey, n_data, axis=0)

	start_time = time.time()
	for i_batch in range(n_batches):
		start = i_batch * batch_size 
		end = start + batch_size
		batch = jax.tree_map(lambda x : x[shuffled_inds[start:end]], training_data)

		hbatch = jax.tree_map(lambda x : x[hshuffled_inds[start:end]], training_data)
		if hbatch_size < batch_size:
			hbatch = {'m':hbatch['m'][:hbatch_size],'u':hbatch['u'][:hbatch_size]}
		params = optimizer.update(params,batch)

	# Remainder
	if remainder > 0:
		batch = jax.tree_map(lambda x : x[shuffled_inds[end:end+remainder]], training_data)

		params = optimizer.update(params,batch)



	epoch_time = time.time() - start_time
	print('The last epoch took', epoch_time, 's')

	metric_t0 = time.time()
	# Post-process and compute metrics after each epoch
	metrics_train = compute_metrics(state,params,training_data)
	metric_time = time.time() - metric_t0

	metrics_history['train_loss'].append(np.array(metrics_train['loss']))
	metrics_history['train_accuracy'].append(np.array(metrics_train['acc']))
	metrics_history['epoch_time'].append(epoch_time)
	metrics_history['epoch'].append(epoch)
	metrics_history['train_loss_woreg'].append(np.array(metrics_train['loss_without_reg']))



	print(f"train epoch: {epoch}, "
				f"loss: {metrics_history['train_loss'][-1]}, "
				f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
	print(f"train epoch: {epoch}, "
					f"loss woreg: {metrics_history['train_loss_woreg'][-1]}")
	print('The metrics took', metric_time, 's')



################################################################################
# Post-processing
logger = metrics_history
run_name = method+'_lr'+str(step_size)+'_n'+str(num_epochs)+'_ds'+str(args.run_seed)

if method in ['adahessian','dnewton','dan', 'dan2']:
		run_name += '_k'+str(k_rank)+'_g'+str(gamma_damping)  

logging_dir = 'ijcnn_logistic_logging/'
os.makedirs(logging_dir,exist_ok = True)
logger_name = run_name+'.pkl'

with open(logging_dir+logger_name, 'wb+') as f:
		pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)

print(80*'#')
print('Run finished and saved successfully'.center(80))





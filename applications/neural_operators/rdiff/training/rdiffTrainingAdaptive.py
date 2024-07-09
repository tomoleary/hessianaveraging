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
parser.add_argument('-step_size', '--step_size', type=float, default=1e-3, help="What step size or 'learning rate'?")
parser.add_argument('-k_rank', '--k_rank', type=int, default=1, help="Rank for Hessian approximation")
parser.add_argument('-gamma_damping', '--gamma_damping', type=float, default=1e-4, help="L-M damping")
parser.add_argument('-epsilon', '--epsilon', type=float, default=1e-8, help="What epsilon for adaptive methods")
parser.add_argument('-beta_2', '--beta_2', type=float, default=0.999, help="exponential decay rate in diagonal preconditioner")
parser.add_argument('-hess_frequency', '--hess_frequency', type=int, default=1, help="Hessian update frequency")

parser.add_argument('-num_epochs', '--num_epochs', type=int, default=100, help="How many epochs")

parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")
parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")
parser.add_argument('-regularization', '--regularization', type=float, default=0.0, help="Tikhonov regularization parameter")
parser.add_argument('-dataset_dir', '--dataset_dir', type=str, default='../data/rdiff_data/', help="Where to load the data from")

parser.add_argument('-rb_choice', '--rb_choice', type=str, default='as', help="choose from [as, kle / pca, None]")
parser.add_argument('-rb_dir', '--rb_dir', type=str, default='../reduced_bases/', help="Where to load the reduced bases from")
parser.add_argument('-rb_rank', '--rb_rank', type=int, default=200, help="RB dim")
parser.add_argument('-depth', '--depth', type=int, default=5, help="NN depth")

parser.add_argument('-dino', '--dino', type=int, default=0, help='DINO training?')

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
sys.path.append('../../../../')
sys.path.append(os.environ.get('HESSAVG_PATH'))
from hessavg import Optimizer, Adam, GradientDescent, AdaGrad, RMSProp, AdaHessian, DiagonalNewton,\
					 DiagonallyAveragedNewton

################################################################################
# Run hyperparameters
method = args.method
assert method in ['adam', 'adahessian','adagrad','gd','mgd','dnewton','dan','dan2','rmsprop']
# num_epochs = args.num_epochs
step_size = args.step_size
k_rank = args.k_rank
gamma_damping = args.gamma_damping
epsilon = args.epsilon
# num_epochs = args.num_epochs
batch_size = args.batch_size
hbatch_size = args.hbatch_size
hessian_frequency = args.hess_frequency


if method in ['dan','dan2','dnewton','adahessian']:
		assert hbatch_size <= batch_size


################################################################################
# Get the data
t0 = time.time()

sys.path.append('../')

from rdiffData import get_rdiff_data

if (args.rb_choice.lower() == 'pca') or (args.rb_choice.lower() == 'kle') :
	# The input data are truncated using PCA / active subspace.
	input_projector = np.load(args.rb_dir+'KLE_projector.npy')[:,:args.rb_rank]
	inpuit_basis = np.load(args.rb_dir+'KLE_basis.npy')[:,:args.rb_rank]
elif args.rb_choice.lower() == 'as':
	# The input data are truncated using PCA / active subspace.
	input_projector = np.load(args.rb_dir+'AS_input_projector.npy')[:,:args.rb_rank]
	input_basis = np.load(args.rb_dir+'AS_input_basis.npy')[:,:args.rb_rank]

else:
	input_projector = False

mydata_name = 'data_'+args.rb_choice.lower() +'_'+str(args.rb_rank)+'.npz'
try:
	reduced_data = np.load(args.dataset_dir+mydata_name)
	training_data = {'m':reduced_data['m_train'],'u':reduced_data['u_train']}
	testing_data = {'m':reduced_data['m_test'],'u':reduced_data['u_test']}

	dM = training_data['m'].shape[1]
	dU = training_data['u'].shape[1]

	print('Successfully loaded the reduced data')
except:
	print('Loading reduced data failed')
	training_data, testing_data, dM, dU = get_rdiff_data(args.dataset_dir,\
										input_projector = input_projector, input_basis = input_basis, derivatives = True)

	np.savez(args.dataset_dir+mydata_name,m_train = training_data['m'],u_train =training_data['u'],J_train = training_data['J'],\
							m_test = testing_data['m'],u_test =testing_data['u'],J_test = testing_data['J'])
	print('Saved the reduced data for next time.')

################################################################################
sys.path.append('../../')
from neuralOperators import GenericDense
nn_widths = args.depth*[2*args.rb_rank]

model = GenericDense(layer_widths=nn_widths, activation='gelu', output_size=dU)

rng = jax.random.PRNGKey(args.run_seed)

params_rng, dropout_rng = jax.random.split(rng)

@jax.jit
def initialize(params_rng):
	init_rngs = {'params': params_rng}
	input_shape = (1, dM)
	variables = model.init(init_rngs, jnp.ones(input_shape, jnp.float32))
	return variables

################################################################################
# Initialize weights and define training procedure

variables = initialize(params_rng)
params = variables['params']

rav_param, unravel = ravel_pytree(params)

print('dW = ',rav_param.shape[0])

################################################################################
# Loss function
from metrics import squared_l2_norm, squared_l2_error


def loss_fn(network, params, batch, train = True):
	y_prediction = network.apply({'params': params}, batch['m'])
	y_true = batch['u']
	err = mse(y_true, y_prediction)
	normalization = jnp.mean(jax.vmap(squared_l2_norm)(y_true), axis=0)
	return err/normalization

################################################################################
# Train state only used to make metrics faster

tx = optax.adam(learning_rate=args.step_size)

state = train_state.TrainState.create(
		apply_fn = model.apply,
		params = variables['params'],
		tx = tx)

def mse(y_true_batched, y_pred_batched):
	return jnp.mean(jax.vmap(squared_l2_error)(y_true_batched, y_pred_batched), axis=0)

def compute_batched_errs(state, batch, params):
	variables = {'params': params}
	y_preds = state.apply_fn(variables, batch['m'])
	y_true = batch['u']

	errs = jax.vmap(squared_l2_error)(y_true, y_preds)
	normalizations = jax.vmap(squared_l2_norm)(y_true)

	return errs,normalizations

@jax.jit
def compute_metrics(state, params, data, batch_size = 32):
	losses = []
	accs = []

	n_data = data['m'].shape[0]

	n_batches, remainder = divmod(n_data, batch_size)
	errs = None 
	normalizations = None

	for i_batch in range(n_batches):
		start = i_batch * batch_size 
		end = start + batch_size
		batch = jax.tree_map(lambda x : x[start:end], data) # no shuffling needed here
		errs_i, normalizations_i = compute_batched_errs(state,batch,params)
		if errs is None:
			errs = jnp.copy(errs_i)
			normalizations = jnp.copy(normalizations_i)
		else:
			errs = jnp.concatenate([errs,errs_i])
			normalizations = jnp.concatenate([normalizations,normalizations_i])
	# Remainder
	if remainder > 0:
		batch = jax.tree_map(lambda x : x[end:end+remainder], data)
		errs_i, normalizations_i = compute_batched_errs(state,batch,params)
		errs = jnp.concatenate([errs,errs_i])
		normalizations = jnp.concatenate([normalizations,normalizations_i])

	rel_squared_errors = jnp.divide(errs,normalizations)

	rms_rel_error = jnp.sqrt(jnp.mean(rel_squared_errors,axis = 0))
	acc = 1. - rms_rel_error

	loss = jnp.mean(errs,axis=0)

	return {'acc': acc,'loss': loss}



################################################################################
# Begin instancing optimizer, etc.

loss = lambda params, batch: loss_fn(model, params, batch)

# Adaptive gradient sampling
gbatch_sizes = [32,128,512,2048,8192]

# batch_epochs = [80,5,5,5,5]
batch_epochs = [160,10,10,10,10]

num_epochs = np.sum(batch_epochs)



# LR schedule
n_data = training_data['m'].shape[0]
n_batches, remainder = divmod(n_data, batch_size)

train_steps_per_epoch = n_batches + int(bool(remainder))
num_train_steps = train_steps_per_epoch * num_epochs

if args.lr_schedule == 'piecewise':
	print('Using piecewise lr schedule')
	lr_schedule = optax.piecewise_constant_schedule(init_value=step_size,
						 boundaries_and_scales={int(num_train_steps*0.75):0.1})

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
														 k_rank = k_rank, gamma_damping = gamma_damping,beta_2= args.beta_2)
elif method.lower() == 'dan2':
	optimizer = DiagonallyAveragedNewton(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
														 k_rank = k_rank, gamma_damping = gamma_damping,\
														 norm_exponent = 2,beta_2= args.beta_2)
elif method.lower() == 'gd':
	optimizer = GradientDescent(loss,step_size = step_size)
elif method.lower() == 'mgd':
	optimizer = MomentumGradientDescent(loss,params, lr_schedule = lr_schedule, step_size = step_size)
elif method.lower() == 'rmsprop':
	optimizer = RMSProp(loss,params, lr_schedule = lr_schedule, step_size = step_size,\
										weight_decay = weight_decay, gamma= args.beta_2)
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
metrics_train = compute_metrics(state,params,training_data)
metrics_test = compute_metrics(state,params,testing_data)
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

################################################################################
# Training loop

rng_key = jax.random.PRNGKey(0)
hrng_key = jax.random.PRNGKey(1) # a different key for the hessian
# Training loop
epoch_counter = 0
for outer_counter, gbatch_size in enumerate(gbatch_sizes):
	n_batches, remainder = divmod(n_data, gbatch_size)
	print('Commencing training loop for |X_k| =', gbatch_size)
	if method.lower() in ['adahessian','dan', 'dan2']:
		if epoch_counter > 10:
			optimizer.hessian_frequency = hessian_frequency

	for epoch in range(epoch_counter,epoch_counter +batch_epochs[outer_counter]):
		rng_key, subkey = jax.random.split(rng_key)
		shuffled_inds = jax.random.permutation(subkey, n_data, axis=0)
		# Hessian data shuffling
		hrng_key, hsubkey = jax.random.split(hrng_key)
		hshuffled_inds = jax.random.permutation(hsubkey, n_data, axis=0)

		start_time = time.time()
		for i_batch in range(n_batches):
			start = i_batch * gbatch_size 
			end = start + gbatch_size
			batch = jax.tree_map(lambda x : x[shuffled_inds[start:end]], training_data)

			hbatch = jax.tree_map(lambda x : x[hshuffled_inds[start:end]], training_data)
			if hbatch_size < gbatch_size:
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
		metrics_test = compute_metrics(state,params,testing_data)
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
	epoch_counter += batch_epochs[outer_counter]


################################################################################
# Post-processing
logger = metrics_history
run_name = method+'_lr'+str(step_size)+'_n'+str(num_epochs)+'_ds'+str(args.run_seed)

if method in ['adahessian','dnewton','dan', 'dan2']:
	run_name += '_k'+str(k_rank)+'_g'+str(gamma_damping)  

if method in ['adahessian','dan','dan2'] and args.hess_frequency > 1:
	run_name += '_hf'+str(args.hess_frequency)

run_name += 'adaptive'

logging_dir = 'rdiff_l2_logging/'
os.makedirs(logging_dir,exist_ok = True)
logger_name = run_name+'.pkl'

with open(logging_dir+logger_name, 'wb+') as f:
	pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)

print(80*'#')
print('Run finished and saved successfully'.center(80))





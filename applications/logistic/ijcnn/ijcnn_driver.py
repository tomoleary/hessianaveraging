import sys, os
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-method', '--method', type=str, default='fan', help="What method")
parser.add_argument('-averaging', '--averaging', type=str, default='exp', help="What type of averaging")
parser.add_argument('-batch_size', '--batch_size', type = int, default = 100, help = 'gradient batch size')
parser.add_argument('-num_epochs', '--num_epochs', type=int, default=10, help="How many epochs")
parser.add_argument('-beta_2', '--beta_2', type=float, default=0.25, help="Parameter for exponential decay averaging")
parser.add_argument('-line_search', '--line_search', type=int, default=1, help="Armijo line search")

parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")
parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")
parser.add_argument('-dataset_dir', '--dataset_dir', type=str, default='/storage/tom/resnet_stuff/', help="Where to store the data")



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
from hessavg import Optimizer, Adam, GradientDescent, MomentumGradientDescent, AdaGrad,\
					 RMSProp, AdaHessian, DiagonalNewton,\
					 DiagonallyAveragedNewton, FullNewton, FullyAveragedNewton


################################################################################
# Run hyperparameters
method = args.method
assert method.lower() in ['gd','newton','fan','mgd','stoch_gd','stoch_mgd','stoch_newton']
averaging = args.averaging 
assert averaging.lower() in ['exp','uni']
num_epochs = args.num_epochs
batch_size = args.batch_size
beta_2 = args.beta_2
line_search = args.line_search
verbose = False



################################################################################
# Get the data
ijcnn_data = np.load('ijcnn1.npz')
# print(list(ijcnn_data.keys()))
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
# Setup the logistic regression model
sys.path.append('../')
from logistic_utilities import LogisticRegressionModel

model = LogisticRegressionModel()

params = model.init(jax.random.PRNGKey(0), np.zeros((1,input_size)))
rav_param, unravel = ravel_pytree(params)
# rav_param = jnp.zeros_like(rav_param)
# params = unravel(rav_param)
dW = rav_param.shape[0]
print(80*'#')
print('Dimension of weights = ',dW)


################################################################################
# Loss function


def loss_fn(network, params, batch):
	label = jnp.squeeze(batch['label'])
	logits = network.apply(params, batch['image'])
	temp = jnp.multiply(label,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean() + (1./(2*n_data))*jnp.linalg.norm(rav_param)**2
	return loss

################################################################################
# Train state only used to make metrics faster

tx = optax.adam(learning_rate=1e-3)

class TrainState(train_state.TrainState):
	pass

state = TrainState.create(
		apply_fn = model.apply,
		params = params,
		tx = tx)

def compute_loss_acc(params,logits, labels):
	labels = jnp.squeeze(batch['label'])
	temp = jnp.multiply(labels,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean() + (1./(2*n_data))*jnp.linalg.norm(rav_param)**2

	positive = jnp.array(logits)>0.0
	preds = jnp.array(positive, dtype = int)
	label_bool = labels == 1
	acc = jnp.mean(label_bool == preds).mean()
	return loss, acc, losses.mean()

def compute_batched_errs(state, params, batch):
	logits = state.apply_fn(params, batch['image'])
	labels = jnp.squeeze(batch['label'])

	temp = jnp.multiply(labels,logits)
	losses = jnp.log(1. + jnp.exp(-temp))
	rav_param, unravel = ravel_pytree(params)
	loss = losses.mean()  

	reg = (1./(2*n_data))*jnp.linalg.norm(rav_param)**2

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

if 'mgd' in method.lower():
	optimizer = MomentumGradientDescent(loss, params, line_search = line_search)

elif 'gd' in method.lower():
	optimizer = GradientDescent(loss, line_search = line_search)

elif 'newton' in method.lower():
	optimizer = FullNewton(loss, line_search = line_search)

elif method.lower() == 'fan':
	optimizer = FullyAveragedNewton(loss, params, line_search = line_search,\
						 averaging=averaging,beta_2=beta_2)

else:
	raise


################################################################################
# Logging utilities

def create_logger(metrics):
	logger = {}
	for key, val in metrics.items():
		logger[key] = [val]
	return logger

def update_logger(logger,metrics):
	for key in metrics:
		logger[key].append(float(metrics[key]))
	return logger


################################################################################
# Begin instancing optimizer etc.

# Using all data for the gradient for now
batch = training_data

metrics_train = compute_metrics(state,params,training_data)
logger = create_logger(metrics_train)
logger['time'] = [0.0]
logger['w'] = np.array(ravel_pytree(params)[0])
print(80*'#')
print('Initial loss = ',logger['loss'][-1])

sufficient_descent = True

iteration = 0
hrng_key = jax.random.PRNGKey(1) 
n_batches, remainder = divmod(n_data, batch_size)

################################################################################
# Define stochatic loop

def stochastic_iteration(params,batch,sub_batch,logger):
	sufficient_descent = True
	t0 = time.perf_counter()
	if method.lower() in ['stoch_gd','stoch_mgd']:
		if line_search:
			params, sufficient_descent = optimizer.update(params,sub_batch)
		else:
			params = optimizer.update(params,sub_batch)
	elif method.lower() in ['fan','stoch_newton']:
		if line_search:
			params, sufficient_descent = optimizer.update(params,batch,hess_batch = sub_batch)
		else:
			params = optimizer.update(params,batch,hess_batch = sub_batch)
	iteration_time = time.perf_counter() - t0

	metrics_train = compute_metrics(state,params,training_data)
	logger = update_logger(logger,metrics_train)
	logger['time'].append(iteration_time)
	logger['w'] = np.array(ravel_pytree(params)[0])
	return params, logger, sufficient_descent

################################################################################
# Commence optimization

for epoch in range(1,num_epochs):
	print('epoch = ',epoch)
	# Deterministic methods
	if method.lower() in ['newton','gd','mgd']:
		t0 = time.perf_counter()
		if line_search:
			params, sufficient_descent = optimizer.update(params,batch)
		else:
			params = optimizer.update(params,batch)
		iteration_time = time.perf_counter() - t0
		# Post-process and report
		# print('params = ',params)
		# print('training_data = ', training_data)
		metrics_train = compute_metrics(state,params,training_data)
		logger = update_logger(logger,metrics_train)
		logger['time'].append(iteration_time)
		logger['w'] = np.array(ravel_pytree(params)[0])
		iteration +=1

		print('iteration = ',iteration)
		print('loss = ',logger['loss'][-1])
		if not sufficient_descent:
			print('Optimization terminated due to Armijo')
			break

	elif method in ['stoch_gd','stoch_mgd','stoch_newton','fan']:
		# Subsampled methods
		if not sufficient_descent:
			print('Optimization terminated due to Armijo')
			break
		print('Subsampled method')
		# A new shuffle of indices for this epoch
		hrng_key, hsubkey = jax.random.split(hrng_key)
		hshuffled_inds = jax.random.permutation(hsubkey, n_data, axis=0)

		for i_batch in range(n_batches):
			start = i_batch * batch_size 
			end = start + batch_size
			sub_batch = jax.tree_map(lambda x : x[hshuffled_inds[start:end]], training_data)

			params, logger, sufficient_descent = stochastic_iteration(params,batch,sub_batch,logger)

			iteration +=1
			if verbose:
				print('iteration = ',iteration)
				print('loss = ',logger['loss'][-1])
			# # print(logger)

			if not sufficient_descent:
				print('Optimization terminated due to Armijo')
				break

		if remainder >0:
			sub_batch = jax.tree_map(lambda x : x[hshuffled_inds[end:end+remainder]], training_data)
			params, logger, sufficient_descent = stochastic_iteration(params,batch,sub_batch,logger)

			iteration +=1
			if verbose:
				print('iteration = ',iteration)
				print('loss = ',logger['loss'][-1])

print('loss = ',logger['loss'][-1])
print('Reached the end of the training procedure')

################################################################################
# Logging
for key in logger:
	logger[key] = np.array(logger[key])

# Name the run
run_name = method+'_n'+str(args.num_epochs)
if method in ['stoch_gd','stoch_newton','fan']:
	run_name += '_batch'+str(batch_size)
if method == 'fan':
	run_name += '_ave'+averaging
	if averaging == 'exp':
		run_name += '_beta2'+str(beta_2)
run_name += str(args.run_seed)

logging_dir = 'ijcnn_logistic_logging/'
os.makedirs(logging_dir,exist_ok = True)
logger_name = run_name+'.pkl'

with open(logging_dir+logger_name, 'wb+') as f:
		pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)




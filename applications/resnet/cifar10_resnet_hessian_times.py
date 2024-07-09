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


parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")
parser.add_argument('-dataset_dir', '--dataset_dir', type=str, default='/storage/tom/resnet_stuff/', help="Where to store the data")
parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")

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
from jax import random
from jax import grad, jit, vmap, jacobian, hessian

from flax import linen as nn
from flax.training import train_state, common_utils
import optax

import pickle
jax.local_devices()

################################################################################
# Get the data
t0 = time.time()

IMAGE_SIZE = 32
NUM_CLASSES = 10
n_outputs = NUM_CLASSES
sys.path.append('../')
from dataUtilities import *

train_set, test_set = get_cifar10_dataset(path_prefix = args.dataset_dir)

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
batch_stats = variables['batch_stats']

################################################################################
# Loss function

def loss_fn(network, params, batch, batch_stats = None,train = True):
		assert batch_stats is not None
		logits, batch_stats = network.apply({'params': params, 'batch_stats': batch_stats}, batch['image'],\
						rngs={'dropout':jax.random.PRNGKey(0)}, mutable = ['batch_stats'],train = train)
		one_hot = jax.nn.one_hot(batch['label'], n_outputs)
		loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
		return loss, batch_stats

batch_stats = variables['batch_stats']
loss = lambda params, batch, batch_stats: loss_fn(model, params, batch, batch_stats)


################################################################################
# Gradient and Hessian vector product timings.

rav_param, unravel = ravel_pytree(params)
rng_key = random.PRNGKey(0)
d = rav_param.shape[0]
all_time_data = {}


@partial(jit,static_argnames=['data_generator'])
def time_gradients_and_hessians(data_generator,batch_stats):
	time_data = {'H':{}}
	the_data = next(iter(data_generator))

	batch = {'image':the_data[0],'label':the_data[1]}

	rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)

	# Gradient
	t0_g0 = time.time()
	grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
	tg0 = time.time() - t0_g0
	time_data['g0'] = tg0
	print('Gradient time0 took ',tg0,'s')

	t0_g = time.time()
	grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
	tg = time.time() - t0_g
	time_data['g'] = tg
	print('Gradient time took ',tg,'s')

	# Hessians
	def Hmp(primal,tangent_matrix):
			gmp = lambda primal: jnp.einsum('i,ij->j',grad(rav_batch_loss,has_aux = True)(primal)[0],tangent_matrix)
			Hmp = jacobian(gmp)
			return Hmp(primal).T

	t0_H0 = time.time()
	Omega = random.rademacher(rng_key, (d,1))
	HOmega = Hmp(rav_param,Omega)
	tH0 = time.time() - t0_H0
	print('H0 took ',tH0,'s')
	time_data['tH0'] = tH0

	for k_rank in [1,5,10,20,30,40,50,60,70,80,90,100]:
		t0_Hk = time.time()
		Omega = random.rademacher(rng_key, (d,k_rank))
		HOmega = Hmp(rav_param,Omega)
		tHk = time.time() - t0_Hk
		time_data['H'][k_rank] = tHk
		print('Hessian with rank',k_rank,' took ',tHk,'s')
		print('HOmega.shape = ',HOmega.shape)

	return time_data, batch_stats


batch_sizes = [32,64,128,256,512,1024,2048]

for batch_size in batch_sizes:
	torch.cuda.empty_cache()
	print(80*'#')
	print('Running for batch size = ',batch_size)
	data_generator = data.DataLoader(test_set,
										 batch_size=batch_size,
										 shuffle=True,
										 drop_last=True,
										 collate_fn=numpy_collate,
										 num_workers=1,
										 persistent_workers=True,
										 worker_init_fn = np.random.seed(args.run_seed))

	all_time_data[str(batch_size)], batch_stats = time_gradients_and_hessians(data_generator, batch_stats)

from datetime import datetime
today = datetime.today().strftime('%Y-%m-%d')

logging_dir = 'cifar10_resnet_logging/'
os.makedirs(logging_dir,exist_ok = True)
logger_name = 'g_and_H_timings'+str(args.run_seed)+'.pkl'

with open(logging_dir+logger_name, 'wb+') as f:
		pickle.dump(all_time_data, f, pickle.HIGHEST_PROTOCOL)




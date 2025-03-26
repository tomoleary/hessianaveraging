# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from functools import partial

import jax
from jax.flatten_util import ravel_pytree

import jax.numpy as jnp

# # @jax.jit
# def tree_stack(*trees):
#     return jax.tree_map(lambda *v: jnp.stack(v), *trees)

def tree_stack(dict1,dict2):
	# for key in dict2:
	# 	print('key = ', key )
	# 	print('dict1[key] = ',type(dict1[key]))
	# 	print('dict2[key] = ',type(dict2[key]))
	# 	print('stacked = ',jnp.stack(dict1[key],dict2[key]))

	# exit()

	for key in dict2:
		dict1[key] = jnp.concatenate((dict1[key],dict2[key]))
	return dict1

class NormTestGradientSampler:


	def __init__(self, loss,data,theta = 0.5, n0 = 1, increment = 1, seed = 0, component_loss = None,\
						start_index = 0, approximate_norm_test = False, verbose = False):

		self.loss = loss
		self.theta = theta

		self.data = data

		self.n_data = len(self.data[list(self.data.keys())[0]])
		self.start_index = start_index

		self.n0 = n0

		self.increment = increment

		assert (self.n_data - self.n0)% self.increment == 0, 'Batch size and increment need to agree with total cardinality'

		self.rng_key = jax.random.PRNGKey(seed)

		self.approximate_norm_test = approximate_norm_test
		self.component_loss = component_loss
		if self.approximate_norm_test:
			assert self.component_loss is not None

		self.verbose = verbose

		self.full_data = False

		self.reshuffle(False)

	def reshuffle(self, deterministic = True):
		# A new shuffle of indices for this epoch

		# In order to avoid repeat samples in a batch for now we need new logic
		# For now simply avoiding this issue by fixing the samples deterministically
		if not deterministic:
			self.rng_key, subkey = jax.random.split(self.rng_key)
			self.shuffled_inds = jax.random.permutation(subkey, self.n_data, axis=0)
		return self.shuffled_inds


	def sample(self,params, n0 = None, batch_stats = None, theta_k = None):
		"""
		n0: initial batch size choice
		"""
		if theta_k is None:
			theta_k = self.theta

		if n0 is None:
			n0 = self.n0

		print('n0 = ',n0)

		if self.full_data:
			return self.data
		else:
			sub_batch, n_chosen, start_index =  sample_for_norm_test(self.loss, params, self.data,\
										theta = theta_k, n0 = n0, increment = self.increment,\
										start_index = self.start_index,\
										index_map = self.shuffled_inds, batch_stats = batch_stats,\
										reshuffle_fn= self.reshuffle, verbose = self.verbose,\
										approximate_norm_test = self.approximate_norm_test,\
										component_loss = self.component_loss)
			self.n0 = n_chosen
			assert n_chosen <= self.n_data
			if n_chosen == self.n_data:
				self.full_data = True
				return self.data
			print('n_chosen = ',n_chosen)

			# Prepare for the next iteration
			self.start_index = start_index
			return sub_batch 



def sample_for_norm_test(loss, params, data, theta, n0, increment,\
					 start_index, index_map, batch_stats, reshuffle_fn, verbose = False,\
					 approximate_norm_test = False, component_loss = None,rev_mode = True):
	"""
	n0  is the initial batch size
	"""
	# Preparation
	rav_param, unravel = ravel_pytree(params)
	n_data = len(data[list(data.keys())[0]])

	if not approximate_norm_test:
		# True gradient
		if batch_stats is not None:
			true_cost_fn = lambda rw : loss(unravel(rw),data,batch_stats)
		else:
			true_cost_fn = lambda rw : loss(unravel(rw),data)

		true_grad_fn = jax.jit(jax.grad(true_cost_fn))

		true_grad = true_grad_fn(rav_param)

		true_grad_norm = jnp.linalg.norm(true_grad)

	# @partial(jax.jit,static_argnames = ['start_index','increment','n_data','reshuffle_fn'])
	def get_next_batch(start_index,index_map, increment, n_data, data,reshuffle_fn):
		# Check if the requested amount exhuasts the available data
		if start_index + increment == n_data:
			# print('start + increment == n_data')
			if verbose:
				print("requested data exactly equal to amount left")
			batch = jax.tree_map(lambda x : x[index_map[start_index:start_index+increment]], data)
			new_index_map = reshuffle_fn()
			new_start_index = 0

		elif start_index + increment > n_data:
			# print('start_index + increment > n_data')
			if verbose:
				print("requested data exceeds amount available, reshuffling")
			# Get reamining data:
			sub_data0 = jax.tree_map(lambda x : x[index_map[start_index:n_data]], data)
			new_start_index = start_index +increment - n_data
			new_index_map = reshuffle_fn()
			sub_data1 = jax.tree_map(lambda x : x[index_map[0:new_start_index]], data)
			batch = tree_stack(sub_data0,sub_data1)
		else:
			# print('start_index + increment < n_data')
			if verbose:
				print("requested data inside of the the amount available")
			batch = jax.tree_map(lambda x : x[index_map[start_index:start_index+increment]], data)
			new_start_index = start_index + increment
			new_index_map = index_map

		return batch, new_start_index, new_index_map

	norm_condition_met = False

	batch = None

	iteration = 0 
	while not norm_condition_met:
		# print('iteration = ',iteration)
		iteration += 1
		if batch is None:
			new_batch, start_index, index_map = get_next_batch(start_index,index_map, n0, n_data, data,reshuffle_fn)
		else:
			new_batch, start_index, index_map = get_next_batch(start_index,index_map, increment, n_data, data,reshuffle_fn)
		if batch is None:
			batch = new_batch
		else:
			batch = tree_stack(batch,new_batch)
		if verbose:
			print('n_chosen = ',len(batch[list(batch.keys())[0]]))
		# Batch gradient
		if batch_stats is not None:
			batch_cost_fn = lambda rw : loss(unravel(rw),batch,batch_stats)
		else:
			batch_cost_fn = lambda rw : loss(unravel(rw),batch)
		batch_gradient_fn = jax.jit(jax.grad(batch_cost_fn))

		batch_gradient = batch_gradient_fn((rav_param))

		if approximate_norm_test:
			raise
			# Component batch gradient
			# if batch_stats is not None:
			# 	batch_component_cost_fn = lambda rw : component_loss(unravel(rw),batch,batch_stats)
			# else:
			# 	batch_component_cost_fn = lambda rw : component_loss(unravel(rw),batch)
			# if rev_mode:
			# 	batch_gradient_component_fn = jax.jit(jax.jacfwd(batch_component_cost_fn, argnums=0))
			# 	jac_eval = batch_gradient_component_fn(rav_param)
				
			# else:
			# 	batch_gradient_component_fn = jax.jit(jax.jacfwd(batch_component_cost_fn, argnums=1))

			# batch_gradient = batch_gradient_fn((rav_param))
			# batch_gradient_norm = jnp.linalg.norm(batch_gradient)

			# batch_component_gradient = batch_gradient_component_fn((rav_param))
			# if False:
				
			# 	batch_variance = jnp.var(batch_component_gradient,axis = 1)
			# 	# batch_variance /= len(batch[list(batch.keys())[0]])
			# 	approx_error = jnp.sqrt(batch_variance.mean())
			# else:
			# 	batch_component_gradient_norms = jnp.linalg.norm(batch_component_gradient,axis=1)
			# 	print('batch_component_gradient norm = ',batch_component_gradient_norms)
			# 	print('batch_gradient_norm = ',batch_gradient_norm)

			# exit()

			# norm_condition_met = (approx_error < theta*batch_gradient_norm)


		else:
			error = jnp.linalg.norm(batch_gradient - true_grad)
			norm_condition_met = (error < theta*true_grad_norm)

		if verbose:
			if approximate_norm_test:
				print('error = ',approx_error)
				print('theta||g|| = ',theta*batch_gradient_norm)
				print('norm condition met = ',norm_condition_met)
			else:
				print('error = ',error)
				print('theta||g|| = ',theta*true_grad_norm)
				print('norm condition met = ',norm_condition_met)

	n_chosen = len(batch[list(batch.keys())[0]])

	print('n_chosen = ',n_chosen)
	print('n_data = ',n_data)

	assert n_chosen <= n_data

	return batch, n_chosen, start_index


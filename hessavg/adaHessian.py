# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import time

from functools import partial

import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian, hessian
from jax.flatten_util import ravel_pytree
from jax import random

from .optimizer import Optimizer

################################################################################

class AdaHessian(Optimizer):

	def __init__(self,loss, params, lr_schedule = None, step_size = 1e-3, k_rank = 1,\
					beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-4,\
					take_abs = True, weight_decay = None, hessian_frequency = 1):
		"""
		beta_2 is the exponential decay rate for averaging of the diagonal rescaling,
		as in Adam


		https://arxiv.org/pdf/2006.00719.pdf
		"""
		self.loss = loss
		self.step_size = step_size
		self.k_rank = k_rank
		self.rng_key = random.PRNGKey(0)
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.take_abs = take_abs

		self.hessian_frequency = hessian_frequency

		rav_param, unravel = ravel_pytree(params)
		self.m = jnp.zeros_like(rav_param)
		self.v = jnp.zeros_like(rav_param)

		self.iteration = 1

		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(AdaHessian,self).__init__(loss,lr_schedule)

	def update(self,params, batch, hess_batch = None, batch_stats = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1 
		h_iteration, remainder = divmod(self.iteration,self.hessian_frequency)
		update_hessian = (remainder == 0)

		if batch_stats is None:
			params, self.m, self.v = adaHessian_update(self.loss,params, batch,\
										self.m, self.v, self.iteration, h_iteration, hess_batch = hess_batch, rng_key = self.rng_key,\
										step_size = step_size, k_rank = self.k_rank,\
										beta_1 = self.beta_1,beta_2 = self.beta_2,epsilon = self.epsilon,\
										take_abs = self.take_abs,weight_decay = self.weight_decay, update_hessian = update_hessian)
			return params
		else:
			params, self.m, self.v, batch_stats = adaHessian_update(self.loss,params, batch,\
										self.m, self.v, self.iteration, h_iteration, hess_batch = hess_batch, rng_key = self.rng_key,\
										step_size = step_size, k_rank = self.k_rank,\
										beta_1 = self.beta_1,beta_2 = self.beta_2,epsilon = self.epsilon,\
										take_abs = self.take_abs, batch_stats = batch_stats,\
										weight_decay = self.weight_decay, update_hessian = update_hessian)
			return params, batch_stats

@partial(jit,static_argnames=['loss','k_rank','take_abs','update_hessian'])
def adaHessian_update(loss, params, batch, m, v, iteration,h_iteration,\
						   hess_batch = None, rng_key = None,k_rank = 1,random_diagonal = True,\
						   step_size = 1e-4,beta_1 = 0.9,beta_2 = 0.999,epsilon = 1e-8,\
						   take_abs = True, batch_stats = None,weight_decay = None, update_hessian = True):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		# gradient computation
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)

		# Hessian computation
		if update_hessian:
			if hess_batch is None:
				hrav_batch_loss = rav_batch_loss
			else:
				hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch,batch_stats)

			def Hmp(primal,tangent_matrix):
				gmp = lambda primal: jnp.einsum('i,ij->j',grad(hrav_batch_loss,has_aux = True)(primal)[0],tangent_matrix)
				Hmp = jacobian(gmp)
				return Hmp(primal).T

	else:
		# gradient computation
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)

		# Hessian computation
		if update_hessian:
			if hess_batch is None:
				hrav_batch_loss = rav_batch_loss
			else:
				hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch)

			def Hmp(primal,tangent_matrix):
				gmp = lambda primal: jnp.einsum('i,ij->j',grad(hrav_batch_loss)(primal),tangent_matrix)
				Hmp = jacobian(gmp)
				return Hmp(primal).T

	if update_hessian:
		if rng_key is None:
			rng_key = random.PRNGKey(0)
		d = rav_param.shape[0]
		Omega = random.rademacher(rng_key, (d,k_rank))
		HOmega = Hmp(rav_param,Omega)
		numerator = jnp.sum(jnp.multiply(Omega,HOmega),axis = -1)
		denominator = jnp.sum(jnp.multiply(Omega,Omega),axis = -1)
		hdiagonal = jnp.divide(numerator,denominator)
		if take_abs:
			hdiagonal = jnp.abs(hdiagonal)
		v = beta_2*v + (1-beta_2)*jnp.multiply(hdiagonal,hdiagonal)

	m = beta_1*m + (1-beta_1)*grads
	
	m_hat = m/(1. - beta_1**iteration)
	v_hat = v/(1. - beta_2**h_iteration)

	p = jnp.divide(m_hat,jnp.sqrt(v_hat)+epsilon*jnp.ones_like(v_hat))

	if weight_decay is not None:
		p += weight_decay*rav_param

	rav_param -= step_size *p
	if batch_stats is not None:
		return unravel(rav_param), m, v, batch_stats
	else:
		return unravel(rav_param), m, v


	
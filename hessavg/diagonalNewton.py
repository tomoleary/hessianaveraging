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

class DiagonalNewton(Optimizer):

	def __init__(self,loss, lr_schedule = None, step_size = 1e-3, k_rank = 1,\
				 gamma_damping = 1e-4, weight_decay = None):
		self.loss = loss
		self.step_size = step_size
		self.k_rank = k_rank
		self.gamma_damping = gamma_damping
		self.rng_key = random.PRNGKey(0)
		self.iteration = 1
		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay
		super(DiagonalNewton,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, hess_batch = None, batch_stats = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1

		
		return diagonal_Newton_update(self.loss,params,batch,rng_key = self.rng_key,\
									step_size = step_size, k_rank = self.k_rank,\
									gamma_damping = self.gamma_damping, batch_stats = batch_stats,\
									weight_decay = self.weight_decay)

@partial(jit,static_argnames=['loss','k_rank'])
def diagonal_Newton_update(loss, params, batch,hess_batch = None,\
						   rng_key = None,k_rank = 1,\
						   gamma_damping = 1e-4,step_size = 1e-4, batch_stats = None,\
							weight_decay = None):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
		if hess_batch is not None:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch,batch_stats)
		else:
			hrav_batch_loss = rav_batch_loss

		def Hmp(primal,tangent_matrix):
			gmp = lambda primal: jnp.einsum('i,ij->j',grad(hrav_batch_loss,has_aux = True)(primal)[0],tangent_matrix)
			Hmp = jacobian(gmp)
			return Hmp(primal).T
	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		if hess_batch is not None:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch)
		else:
			hrav_batch_loss = rav_batch_loss

		def Hmp(primal,tangent_matrix):
			gmp = lambda primal: jnp.einsum('i,ij->j',grad(hrav_batch_loss)(primal),tangent_matrix)
			Hmp = jacobian(gmp)
			return Hmp(primal).T

	# Diagonal computation
	if rng_key is None:
		rng_key = random.PRNGKey(0)
	d = rav_param.shape[0]
	Omega = random.rademacher(rng_key, (d,k_rank))
	HOmega = Hmp(rav_param,Omega)
	numerator = jnp.sum(jnp.multiply(Omega,HOmega),axis = -1)
	denominator = jnp.sum(jnp.multiply(Omega,Omega),axis = -1)
	hdiagonal = jnp.divide(numerator,denominator)
		   
	diagonal = gamma_damping*jnp.ones_like(hdiagonal) + jnp.abs(hdiagonal)
	p = jnp.true_divide(grads,diagonal)

	if weight_decay is not None:
		p += weight_decay*rav_param

	rav_param -= step_size *p
	if batch_stats is not None:
		return unravel(rav_param), batch_stats
	else:
		return unravel(rav_param)


################################################################################

class DiagonallyAveragedNewton(Optimizer):

	def __init__(self,loss, params, lr_schedule = None,  step_size = 1e-3, k_rank = 1, gamma_damping = 1e-4,\
					beta_2 = 0.999, weight_decay = None, norm_exponent = 1,hessian_frequency = 1):
		"""
		beta_2 is the exponential decay rate for averaging of the diagonal rescaling,
		as in Adam
		"""
		self.loss = loss
		self.step_size = step_size
		self.k_rank = k_rank
		self.gamma_damping = gamma_damping
		self.rng_key = random.PRNGKey(0)

		rav_param, unravel = ravel_pytree(params)
		self.hd = jnp.zeros_like(rav_param)
		self.beta_2 = beta_2

		self.iteration = 1
		self.norm_exponent = norm_exponent

		self.hessian_frequency = hessian_frequency

		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(DiagonallyAveragedNewton,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, hess_batch = None, batch_stats=None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size

		self.iteration += 1 
		h_iteration, remainder = divmod(self.iteration,self.hessian_frequency)
		update_hessian = (remainder == 0)

		if batch_stats is None:

			params, self.hd = diagonally_averaged_Newton_update(self.loss,params,batch,\
										self.hd, h_iteration, hess_batch = hess_batch, rng_key = self.rng_key,\
										step_size = step_size, k_rank = self.k_rank,\
										gamma_damping = self.gamma_damping,beta_2 = self.beta_2,\
										weight_decay = self.weight_decay,norm_exponent = self.norm_exponent,\
										update_hessian = update_hessian)
			return params
		else:
			params, self.hd, batch_stats = diagonally_averaged_Newton_update(self.loss,params,batch,\
										self.hd, h_iteration, hess_batch = hess_batch, rng_key = self.rng_key,\
										step_size = step_size, k_rank = self.k_rank,\
										gamma_damping = self.gamma_damping,beta_2 = self.beta_2,\
										batch_stats = batch_stats,\
										weight_decay = self.weight_decay, norm_exponent = self.norm_exponent,\
										update_hessian = update_hessian)
			return params, batch_stats


@partial(jit,static_argnames=['loss','k_rank','take_abs','norm_exponent','update_hessian'])
def diagonally_averaged_Newton_update(loss, params, batch, hd, h_iteration,\
						   hess_batch = None, rng_key = None,k_rank = 1,\
						   gamma_damping = 1e-4,step_size = 1e-4,beta_2 = 0.999,\
						   take_abs = True,batch_stats = None,weight_decay = None,\
						   norm_exponent = 1,update_hessian = True):
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

		# For now doing exponentially decaying sum
		# may decide to use a different rule later

		# Here update the rule to allow for different averaging and norm exponent

		if norm_exponent == 1:
			hd = beta_2*hd + (1-beta_2)*hdiagonal
			d_hat = hd/(1. - beta_2**h_iteration)
			diagonal = gamma_damping*jnp.ones_like(hdiagonal) + jnp.abs(d_hat)
		elif norm_exponent == 2:
			hd = beta_2*hd + (1-beta_2)*jnp.multiply(hdiagonal,hdiagonal)
			d_hat = hd/(1. - beta_2**h_iteration)
			diagonal = gamma_damping*jnp.ones_like(hdiagonal) + jnp.sqrt(d_hat)
		else:
			raise
	else:
		if norm_exponent == 1:
			d_hat = hd/(1. - beta_2**h_iteration)
			diagonal = gamma_damping*jnp.ones_like(hd) + jnp.abs(d_hat)
		elif norm_exponent == 2:
			d_hat = hd/(1. - beta_2**h_iteration)
			diagonal = gamma_damping*jnp.ones_like(hd) + jnp.sqrt(d_hat)
		else:
			raise


	p = jnp.true_divide(grads,diagonal)
	if weight_decay is not None:
		p += weight_decay*rav_param
	
	rav_param -= step_size *p
	if batch_stats is not None:
		return unravel(rav_param), hd, batch_stats
	else:
		return unravel(rav_param), hd




	
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
from .globalization import armijo_line_search



################################################################################

class FullNewton(Optimizer):

	def __init__(self,loss, lr_schedule = None, step_size = 1e-3, k_rank = 1,\
				 gamma_damping = 1e-4, weight_decay = None,saddle_free = True, line_search = False):
		self.loss = loss
		self.step_size = step_size
		self.gamma_damping = gamma_damping


		self.rng_key = random.PRNGKey(0)
		self.iteration = 1
		self.saddle_free = saddle_free
		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		self.line_search = line_search

		super(FullNewton,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, hess_batch = None, batch_stats = None, full_data = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1

		
		updates = full_Newton_update(self.loss,params,batch, hess_batch = hess_batch,\
									step_size = step_size, gamma_damping = self.gamma_damping,\
									batch_stats = batch_stats,weight_decay = self.weight_decay,\
									saddle_free = self.saddle_free,line_search = self.line_search)

		if self.line_search:
			if batch_stats is not None:
				p, batch_stats, pTg = updates
			else:
				p, pTg = updates
			if full_data is None:
				return armijo_line_search(self.loss,params,p,pTg,batch,batch_stats)
			else:
				return armijo_line_search(self.loss,params,p,pTg,full_data,batch_stats)
		else:
			return updates

@partial(jit,static_argnames=['loss','saddle_free','line_search'])
def full_Newton_update(loss, params, batch,hess_batch = None,\
						   gamma_damping = 1e-4,step_size = 1e-4, batch_stats = None,\
							weight_decay = None,saddle_free = True,line_search = False):
	rav_param, unravel = ravel_pytree(params)
	dW = rav_param.shape[0]
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
		if hess_batch is not None:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch,batch_stats)
		else:
			hrav_batch_loss = rav_batch_loss
		Hess, batch_stats = hessian(hrav_batch_loss,has_aux = True)(rav_param)

	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		if hess_batch is not None:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch)
		else:
			hrav_batch_loss = rav_batch_loss
		Hess = hessian(hrav_batch_loss)(rav_param)

	if saddle_free:
		d,U = jnp.linalg.eigh(Hess)
		Hess = U@jnp.diag(jnp.abs(d))@U.T

	Hess += gamma_damping*jnp.eye(dW)
	p = jnp.linalg.inv(Hess)@grads

	if weight_decay is not None:
		p += weight_decay*rav_param

	# rav_param -= step_size *p
	# if batch_stats is not None:
	# 	return unravel(rav_param), batch_stats
	# else:
	# 	return unravel(rav_param)

	if line_search:
		if batch_stats is not None:
			return unravel(-p), batch_stats, -jnp.inner(p,grads)
		else:
			return unravel(-p), -jnp.inner(p,grads)

	else:
		rav_param -= step_size *p
		if batch_stats is not None:
			return unravel(rav_param), batch_stats
		else:
			return unravel(rav_param)


################################################################################

class FullyAveragedNewton(Optimizer):

	def __init__(self,loss, params, lr_schedule = None,  step_size = 1e-3, gamma_damping = 0.0,\
					beta_2 = 0.999, weight_decay = None, saddle_free = True, line_search = False,\
					averaging = 'exp'):
		"""
		beta_2 is the exponential decay rate for averaging of the diagonal rescaling,
		as in Adam
		"""
		self.loss = loss
		self.step_size = step_size
		self.gamma_damping = gamma_damping
		self.rng_key = random.PRNGKey(0)

		rav_param, unravel = ravel_pytree(params)
		dW = rav_param.shape[0]
		self.Mk = jnp.zeros((dW,dW))
		self.beta_2 = beta_2

		self.iteration = 1

		self.saddle_free = saddle_free
		self.line_search = line_search
		assert averaging in ['exp','uni']
		self.averaging = averaging

		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(FullyAveragedNewton,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, hess_batch = None, batch_stats=None, full_data = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size

		self.iteration += 1

		updates = fully_averaged_Newton_update(self.loss,params,batch,\
								self.Mk, self.iteration, hess_batch = hess_batch, \
								step_size = step_size, saddle_free = self.saddle_free,\
								gamma_damping = self.gamma_damping,beta_2 = self.beta_2,\
								weight_decay = self.weight_decay,line_search=self.line_search,\
								averaging = self.averaging)

		if self.line_search:
			if batch_stats is not None:
				p, self.Mk, batch_stats, pTg = updates
			else:
				p, self.Mk, pTg = updates
			if full_data is None:
				return armijo_line_search(self.loss,params,p,pTg,batch,batch_stats)
			else:
				return armijo_line_search(self.loss,params,p,pTg,full_data,batch_stats)
		else:
			if batch_stats is not None:
				params, self.Mk, batch_stats = updates
				return params, batch_stats
			else:
				params, self.Mk = updates
				return params

		# if batch_stats is None:

		# 	params, self.Mk = fully_averaged_Newton_update(self.loss,params,batch,\
		# 								self.Mk, self.iteration, hess_batch = hess_batch, \
		# 								step_size = step_size, \
		# 								gamma_damping = self.gamma_damping,beta_2 = self.beta_2,\
		# 								weight_decay = self.weight_decay)
		# 	return params
		# else:
		# 	params, self.Mk, batch_stats = fully_averaged_Newton_update(self.loss,params,batch,\
		# 								self.Mk, self.iteration, hess_batch = hess_batch, \
		# 								step_size = step_size, \
		# 								gamma_damping = self.gamma_damping,beta_2 = self.beta_2,\
		# 								batch_stats = batch_stats,\
		# 								weight_decay = self.weight_decay)
		# 	return params, batch_stats


# @partial(jit,static_argnames=['loss', 'saddle_free','line_search','averaging'])
def fully_averaged_Newton_update(loss, params, batch, Mk, iteration,\
						   hess_batch = None, rng_key = None,k_rank = 1,\
						   saddle_free = True,gamma_damping = 1e-4,step_size = 1e-4,\
						   beta_2 = 0.999,batch_stats = None,weight_decay = None,\
						   line_search = False, averaging = 'exp'):

	rav_param, unravel = ravel_pytree(params)
	dW = rav_param.shape[0]
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
		if hess_batch is None:
			hrav_batch_loss = rav_batch_loss
		else:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch,batch_stats)
		Hess, batch_stats = hessian(hrav_batch_loss,has_aux = True)(rav_param)

	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		if hess_batch is None:
			hrav_batch_loss = rav_batch_loss
		else:
			hrav_batch_loss = lambda rw : loss(unravel(rw),hess_batch)
		Hess = hessian(hrav_batch_loss)(rav_param)

	if saddle_free:
		d,U = jnp.linalg.eigh(Hess)
		Hess = U@jnp.diag(jnp.abs(d))@U.T

	if averaging == 'exp':
		Mk = beta_2*Mk + (1-beta_2)*Hess
		Mkk = Mk/(1.-beta_2**iteration) + gamma_damping*jnp.eye(dW)
	elif averaging == 'uni':
		Mk = ((iteration-1.0)/iteration)*Mk + (1./iteration)*Hess 
		Mkk = Mk
		pass
	else:
		raise

	p = jnp.linalg.inv(Mkk)@grads

	if weight_decay is not None:
		p += weight_decay*rav_param

	if line_search:
		if batch_stats is not None:
			return unravel(-p), Mk, batch_stats, -jnp.inner(p,grads)
		else:
			return unravel(-p), Mk, -jnp.inner(p,grads)
	else:
		rav_param -= step_size *p
		if batch_stats is not None:
			return unravel(rav_param), Mk, batch_stats
		else:
			return unravel(rav_param), Mk




	
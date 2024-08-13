# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from functools import partial

import jax.numpy as jnp
from jax import grad, jit, vmap, jacobian, hessian
from jax.flatten_util import ravel_pytree
from jax import random

from .optimizer import Optimizer


################################################################################

class GradientDescent(Optimizer):

	def __init__(self,loss, lr_schedule = None, step_size = 1e-3,\
					weight_decay = None):
		self.loss = loss
		self.step_size = step_size
		self.iteration = 1
		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(GradientDescent,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, hess_batch = None, batch_stats = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1
		return ravel_gd_update(self.loss,params,batch,step_size = step_size,\
				batch_stats = batch_stats, weight_decay = self.weight_decay)


@partial(jit,static_argnames = ['loss'])
def ravel_gd_update(loss, params, batch,step_size = 1e-3,batch_stats = None,\
						weight_decay = None):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss,has_aux = True)(rav_param)
		if weight_decay is not None:
			grads += weight_decay*rav_param
		rav_param -= step_size *grads
		
		return unravel(rav_param), batch_stats
	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		if weight_decay is not None:
			grads += weight_decay*rav_param
		rav_param -= step_size *grads
		return unravel(rav_param)


################################################################################

class MomentumGradientDescent(Optimizer):

	def __init__(self,loss,  params, lr_schedule = None, step_size = 1e-3,\
				 beta = 0.9, weight_decay = None):
		self.loss = loss
		self.step_size = step_size
		self.beta = beta

		# Pass the params into the constructor to figure out 
		# proper array initialization
		rav_param, unravel = ravel_pytree(params)
		self.momentum = jnp.zeros_like(rav_param)
		self.iteration = 1

		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(MomentumGradientDescent,self).__init__(loss, lr_schedule = lr_schedule)


	def update(self,params,batch, batch_stats = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1
		if batch_stats is None:
			params, self.momentum = ravel_momentum_gd_update(self.loss,params,batch, self.momentum,\
												beta = self.beta, step_size = step_size,\
												weight_decay = self.weight_decay)
			return params
		else:
			params, self.momentum, batch_stats = ravel_momentum_gd_update(self.loss,params,batch, self.momentum,\
												beta = self.beta, step_size = step_size,\
												batch_stats = batch_stats, weight_decay = self.weight_decay)
			return params, batch_stats

@partial(jit,static_argnames = ['loss'])
def ravel_momentum_gd_update(loss, params, batch, momentum, beta = 0.9,\
							 step_size = 1e-3,batch_stats = None,weight_decay = None):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss)(rav_param)
		momentum = beta*momentum + grads
		if weight_decay is not None:
			momentum += weight_decay*rav_param
		rav_param -= step_size *momentum
		return unravel(rav_param), momentum, batch_stats

	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		momentum = beta*momentum + grads
		if weight_decay is not None:
			momentum += weight_decay*rav_param
		rav_param -= step_size *momentum
		return unravel(rav_param), momentum



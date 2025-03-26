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
from .globalization import armijo_line_search


################################################################################

class GradientDescent(Optimizer):

	def __init__(self,loss, lr_schedule = None, step_size = 1e-3,\
					weight_decay = None,line_search = False):
		self.loss = loss
		self.step_size = step_size
		self.iteration = 1
		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		self.line_search = line_search

		super(GradientDescent,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, batch_stats = None, full_data = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1
		updates =  ravel_gd_update(self.loss,params,batch,step_size = step_size,\
				batch_stats = batch_stats, weight_decay = self.weight_decay,\
				line_search = self.line_search)
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


@partial(jit,static_argnames = ['loss','line_search'])
def ravel_gd_update(loss, params, batch,step_size = 1e-3,batch_stats = None,\
						weight_decay = None,line_search = False):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss,has_aux = True)(rav_param)
		if weight_decay is not None:
			grads += weight_decay*rav_param

		if line_search:
			return unravel(-1.0*grads), batch_stats, -jnp.inner(grads,grads)
		else:
			rav_param -= step_size *grads
			return unravel(rav_param), batch_stats
	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		if weight_decay is not None:
			grads += weight_decay*rav_param

		if line_search:
			return unravel(-1.0*grads), -jnp.inner(grads,grads)
		else:
			rav_param -= step_size *grads
			return unravel(rav_param)


################################################################################

class MomentumGradientDescent(Optimizer):

	def __init__(self,loss,  params, lr_schedule = None, step_size = 1e-3,\
				 beta = 0.9, weight_decay = None, line_search = False):
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

		self.line_search = line_search

		super(MomentumGradientDescent,self).__init__(loss, lr_schedule = lr_schedule)

	def update(self,params,batch, batch_stats = None, full_data = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1
		# updates =  ravel_gd_update(self.loss,params,batch,step_size = step_size,\
		# 		batch_stats = batch_stats, weight_decay = self.weight_decay,\
		# 		line_search = self.line_search)
		updates = ravel_momentum_gd_update(self.loss,params,batch, self.momentum,\
										beta = self.beta, step_size = step_size,\
										weight_decay = self.weight_decay,
										line_search = self.line_search)
		if self.line_search:
			if batch_stats is not None:
				p, self.momentum, batch_stats, pTg = updates
			else:
				p, self.momentum, pTg = updates

			if full_data is None:
				return armijo_line_search(self.loss,params,p,pTg,batch,batch_stats)
			else:
				return armijo_line_search(self.loss,params,p,pTg,full_data,batch_stats)
		else:
			if batch_stats is not None:
				p, self.momentum, batch_stats = updates
				return p, batch_stats
			else:
				p, self.momentum = updates
				return p



@partial(jit,static_argnames = ['loss','line_search'])
def ravel_momentum_gd_update(loss, params, batch, momentum, beta = 0.9,\
							 step_size = 1e-3,batch_stats = None,weight_decay = None, line_search = False):
	rav_param, unravel = ravel_pytree(params)

	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss,has_aux = True)(rav_param)
		momentum = beta*momentum + grads
		if weight_decay is not None:
			momentum += weight_decay*rav_param

		if line_search:
			return unravel(-1.0*grads), batch_stats, -jnp.inner(grads,grads)
		else:
			rav_param -= step_size *grads
			return unravel(rav_param), momentum, batch_stats
	else:
		rav_batch_loss = lambda rw : loss(unravel(rw),batch)
		grads = grad(rav_batch_loss)(rav_param)
		momentum = beta*momentum + grads
		if weight_decay is not None:
			momentum += weight_decay*rav_param

		if line_search:
			return unravel(-1.0*momentum), momentum, -jnp.inner(momentum,grads)
		else:
			rav_param -= step_size *momentum
			return unravel(rav_param), momentum



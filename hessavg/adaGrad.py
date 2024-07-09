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
# Based on https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad

class AdaGrad(Optimizer):
	def __init__(self,loss, params, lr_schedule = None,\
			 step_size = 1e-3, epsilon = 1e-7,\
			 weight_decay = None):
		self.loss = loss
		self.step_size = step_size
		self.epsilon = epsilon

		# Pass the params into the constructor to figure out 
		# proper array initialization
		rav_param, unravel = ravel_pytree(params)
		self.G = jnp.zeros_like(rav_param)

		self.iteration = 1

		if weight_decay is not None:
			assert type(weight_decay) is float
		self.weight_decay = weight_decay

		super(AdaGrad,self).__init__(loss,lr_schedule = lr_schedule)


	def update(self,params,batch, hess_batch = None, batch_stats = None):
		if self.lr_schedule is not None:
			step_size = self.lr_schedule(self.iteration)
		else:
			step_size = self.step_size
		self.iteration += 1
		if batch_stats is None:
			params, self.G = ravel_adagrad_update(self.loss,params, batch, self.G, self.iteration,\
										step_size = step_size,epsilon = self.epsilon, weight_decay = self.weight_decay)
			return params
		else:
			params, self.G, batch_stats = ravel_adagrad_update(self.loss,params, batch, self.G, self.iteration,\
										step_size = step_size,epsilon = self.epsilon, batch_stats = batch_stats,\
										weight_decay = self.weight_decay)
			return params, batch_stats
		
import numpy as np

@partial(jit,static_argnames = ['loss'])
def ravel_adagrad_update(loss, params, batch, G, iteration,\
		 epsilon = 0.0, step_size = 1e-3, batch_stats = None, weight_decay = None):
	rav_param, unravel = ravel_pytree(params)
	if batch_stats is not None:
		rav_batch_loss = lambda rw : loss(unravel(rw), batch,batch_stats)
		grads, batch_stats = grad(rav_batch_loss, has_aux = True)(rav_param)
	else:	
		rav_batch_loss = lambda rw : loss(unravel(rw), batch)
		grads = grad(rav_batch_loss)(rav_param)

	G = G + jnp.multiply(grads,grads)

	p = jnp.divide(grads,jnp.sqrt(G)+epsilon*jnp.ones_like(G))

	if weight_decay is not None:
		p += weight_decay*rav_param
		
	rav_param -= step_size *p
	if batch_stats is not None:
		return unravel(rav_param), G, batch_stats
	else:
		return unravel(rav_param), G


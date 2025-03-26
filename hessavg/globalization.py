# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import jax
from jax.flatten_util import ravel_pytree


def armijo_line_search(loss,params,p,pTg,batch,batch_stats = None,\
						c_armijo=1e-4,alpha = 1.0,max_backtrack = 20,\
						verbose = False,jit_compile = True):
	'''
	'''
	rav_update, unravel = ravel_pytree(p)
	rav_param, unravel = ravel_pytree(params)
	# Very slow for now
	if batch_stats is not None:
		cost_fn = lambda rw : loss(unravel(rw),batch,batch_stats)
	else:
		cost_fn = lambda rw : loss(unravel(rw),batch)

	if jit_compile:
		cost_fn = jax.jit(cost_fn)

	line_search, line_search_iter = (True,0)
	initial_cost = cost_fn(rav_param)
	# print('pTg = ',pTg)
	# print('||p|| = ', jax.numpy.linalg.norm(rav_update))
	while line_search and  line_search_iter < max_backtrack:
		cost_new = cost_fn(rav_param +alpha*rav_update)
		if verbose:
			print('cost_new', cost_new, 'sufficient descent:', initial_cost + alpha*c_armijo*pTg )
		sufficient_descent = (cost_new < initial_cost + alpha*c_armijo*pTg)
		if sufficient_descent:
			line_search   	= False
		else:
			alpha          *= 0.5
		line_search_iter += 1

	if sufficient_descent:
		rav_param += alpha*rav_update
		if verbose:
			print('alpha = ',alpha)

		# print('alpha = ',alpha)

	if batch_stats is not None:
		return unravel(rav_param), batch_stats, {'sufficient_descent':sufficient_descent,'alpha':alpha}
	else:
		return unravel(rav_param), {'sufficient_descent':sufficient_descent,'alpha':alpha}

	
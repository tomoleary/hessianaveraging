# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import numpy as np
import jax
import jax.numpy as jnp

import os
import pickle

from optimizers import *
from quadratic import *


import argparse

parser = argparse.ArgumentParser(description="Quadratic Minimization tests")
parser.add_argument('-dW', '--dW', type=int, default=100, help="problem dimension")
parser.add_argument('-nA', '--noise_A', type=float, default=0.01, help="relative noise in A")
parser.add_argument('-nb', '--noise_b', type=float, default=0.01, help="relative noise in b")
parser.add_argument('-it', '--iterations', type=int, default=1000, help="total_iterations")
parser.add_argument('-loss', '--loss_function', type=str, default='quadratic', help="Which loss function")
args = parser.parse_args()

A,b = setup_terms(dW = args.dW)
loss = quadratic

# Set up jax data
Aj = jnp.asarray(A)
bj = jnp.asarray(b)

data = default_true_data(loss,Aj,bj)

stochastic_data = default_stochastic_data(loss,Aj,bj)
stochastic_dataH = default_stochastic_data(loss,Aj,bj)

# Setup reference initial guess
x0np = np.random.randn(args.dW)

master_logger = {}
# Gradient Noise test
g = jax.grad(loss)
stochastic_data['random_state'] = np.random.RandomState(seed=0)
stochastic_dataH['random_state'] = np.random.RandomState(seed=1)
grel_noise = []
for i in range(10):
    xgtest = jnp.asarray(np.random.randn(args.dW))
    g_true = g(xgtest,data)
    g_stoch = g(xgtest,stochastic_data)
    grel_noise.append(np.array(jnp.linalg.norm(g_true - g_stoch)/jnp.linalg.norm(g_true)))
master_logger['grel_noise'] = grel_noise


methods = ['sgd','newton', 'sfn','averaged_newton_e','averaged_newton_u','dan_e','dan_u']

method_callables = {'sgd':gradient_descent,\
						'newton':newton,'sfn':newton,\
						'averaged_newton_e':averaged_newton,'averaged_newton_u':averaged_newton,\
						'dan_e':diagonally_averaged_newton, 'dan_u':diagonally_averaged_newton}
method_kwargs = {'sgd':{},\
					'newton':{'saddle_free':False}, 'sfn':{'saddle_free':True},\
					'averaged_newton_e':{'averaging':'exponential'},'averaged_newton_u':{'averaging':'uniform'},\
					'dan_e':{'averaging':'exponential'},'dan_u':{'averaging':'uniform'}}

alphas = [1e0,1e-1,1e-2,1e-3,1e-4,1e-5]
# Run optimizations w/o adaptive gradients
for method in methods:
	master_logger[method] = {}
	for alpha in alphas:
		print('Running ',method, alpha)
		# Get the default data
		stochastic_data = default_stochastic_data(loss,Aj,bj)
		stochastic_dataH = default_stochastic_data(loss,Aj,bj)
		stochastic_data['random_state'] = np.random.RandomState(seed=0)
		stochastic_dataH['random_state'] = np.random.RandomState(seed=1)
		# Set up initial guess
		xopt = jnp.asarray(x0np)
		# Get the kwargs
		kwargs = method_kwargs[method]
		kwargs['alpha'] = alpha
		kwargs['iterations'] = args.iterations
		kwargs['args'] = stochastic_data
		kwargs['true_args'] = data
		if not ('sgd' in method.lower()):
			kwargs['Hargs'] = stochastic_dataH
		xopt, logger_i = method_callables[method](xopt,loss,**kwargs)
		master_logger[method][str(alpha)] = logger_i


# Run optimizations w/ adaptive gradients
for method in methods:
	master_logger[method+'_ag'] = {}
	for alpha in alphas:
		print('Running ',method, alpha)
		# Get the default data
		stochastic_data = default_stochastic_data(loss,Aj,bj)
		stochastic_dataH = default_stochastic_data(loss,Aj,bj)
		stochastic_data['random_state'] = np.random.RandomState(seed=0)
		stochastic_dataH['random_state'] = np.random.RandomState(seed=1)
		# Set up initial guess
		xopt = jnp.asarray(x0np)
		# Get the kwargs
		kwargs = method_kwargs[method]
		kwargs['alpha'] = alpha
		kwargs['iterations'] = args.iterations
		kwargs['args'] = stochastic_data
		kwargs['true_args'] = data
		# With adaptive gradients!!!!!
		kwargs['norm_test'] = quadratic_norm_test
		if not ('sgd' in method.lower()):
			kwargs['Hargs'] = stochastic_dataH
		xopt, logger_i = method_callables[method](xopt,loss,**kwargs)
		master_logger[method+'_ag'][str(alpha)] = logger_i



logger_dir = 'logging/'		
os.makedirs(logger_dir,exist_ok = True)
run_spec = args.loss_function+'_dW'+str(args.dW)+'_nA'+str(args.noise_A)+'_nb'+str(args.noise_b)+'_it'+str(args.iterations)
filename = logger_dir+run_spec+'.pkl'

with open(filename,'wb+') as f:
	pickle.dump(master_logger,f,pickle.HIGHEST_PROTOCOL)


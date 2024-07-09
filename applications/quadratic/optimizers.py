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


# Gradient Descent
def gradient_descent(x,loss,args = None, true_args = None, alpha = 1e-4,iterations=100,\
             print_freq = 10,norm_test = None,theta = 0.5):
    logger = {'cost': []}
    g = jax.grad(loss)
    for k in range(iterations):
        if true_args is None:
            logger['cost'].append(np.array(loss(x,args)))
        else:
            logger['cost'].append(np.array(loss(x,true_args)))
        if np.isnan(logger['cost'][-1]):
            return x, logger
        if k%print_freq == 0 :
            print('Iteration ',k,'cost = ',logger['cost'][-1])
        p = g(x,args)
        x -= alpha*p
        if norm_test is not None:
            args = norm_test(loss,x,args,theta)
    return x, logger

# Generic Newton's Method
def newton(x,loss,args = None,Hargs=None, true_args = None, alpha = 1.0,iterations=100, print_freq = 10,\
            saddle_free = False,norm_test = None,theta = 0.5):
    if Hargs is None:
        Hargs = args
    logger = {'cost': []}
    g = jax.grad(loss)
    H = jax.hessian(loss)
    for k in range(iterations):
        if true_args is None:
            logger['cost'].append(np.array(loss(x,args)))
        else:
            logger['cost'].append(np.array(loss(x,true_args)))
        if np.isnan(logger['cost'][-1]):
            return x, logger
        if k%print_freq == 0 :
            print('Iteration ',k,'cost = ',logger['cost'][-1])
        Hk = H(x,Hargs)
        if saddle_free:
            d,U = jnp.linalg.eigh(Hk)
            Hk = U@jnp.diag(jnp.abs(d))@U.T
        gk = g(x,args)
        p = jnp.linalg.inv(Hk)@gk
        x -= alpha*p
        if norm_test is not None:
            args = norm_test(loss,x,args,theta)
    return x, logger

# Averaged Newton method
def averaged_newton(x,loss,args = None,Hargs = None, true_args = None, alpha = 1.0,beta = 0.99,\
                        iterations=100, print_freq = 10,bias_correction = False,\
                        saddle_free = False,averaging = 'exponential',\
                        norm_test = None,theta = 0.5):
    if Hargs is None:
        Hargs = args
    logger = {'cost': []}
    g = jax.grad(loss)
    H = jax.hessian(loss)
    M = None
    for k in range(iterations):
        if true_args is None:
            logger['cost'].append(np.array(loss(x,args)))
        else:
            logger['cost'].append(np.array(loss(x,true_args)))
        if np.isnan(logger['cost'][-1]):
            return x, logger
        if k%print_freq == 0 :
            print('Iteration ',k,'cost = ',logger['cost'][-1])
        Hk = H(x,Hargs)
        if saddle_free:
            d,U = jnp.linalg.eigh(Hk)
            Hk = U@jnp.diag(jnp.abs(d))@U.T
        if averaging.lower() == 'exponential':
            if M is None:    
                M = Hk
            else:
                M = beta*M + (1. - beta)*Hk
            if bias_correction:
                Mk = M/(1. - beta**k)
            else:
                Mk = M
        elif averaging.lower() == 'uniform':
            if M is None:    
                M = Hk
            else:
                M= (1/k+1)*Hk + k/(k+1)*M
            Mk = M
        gk = g(x,args)
        p = jnp.linalg.inv(Mk)@gk
        x -= alpha*p
        if norm_test is not None:
            args = norm_test(loss,x,args,theta)
    return x, logger



# Diagonally Averaged Newton method
def diagonally_averaged_newton(x,loss,args = None,Hargs = None,true_args = None, alpha = 1.0,beta = 0.99,\
                        iterations=100, print_freq = 10,bias_correction = False,\
                        averaging = 'exponential',gamma_damping=1e-4,\
                        norm_test = None,theta = 0.5):
    if Hargs is None:
        Hargs = args
    logger = {'cost': []}
    g = jax.grad(loss)
    H = jax.hessian(loss)
    M = None
    for k in range(iterations):
        if true_args is None:
            logger['cost'].append(np.array(loss(x,args)))
        else:
            logger['cost'].append(np.array(loss(x,true_args)))
        if np.isnan(logger['cost'][-1]):
            return x, logger
        if k%print_freq == 0 :
            print('Iteration ',k,'cost = ',logger['cost'][-1])
        Hk = jnp.abs(jnp.diagonal(H(x,Hargs)))

        if averaging.lower() == 'exponential':
            if M is None:    
                M = Hk
            else:
                M = beta*M + (1. - beta)*Hk
            if bias_correction:
                Mk = M/(1. - beta**k)
            else:
                Mk = M
        elif averaging.lower() == 'uniform':
            if M is None:    
                M = Hk
            else:
                M= (1/k+1)*Hk + k/(k+1)*M
            Mk = M
        diagonal = gamma_damping*jnp.ones_like(Mk) + jnp.abs(Mk)
        gk = g(x,args)
        p = jnp.true_divide(gk,diagonal)
        x -= alpha*p
        if norm_test is not None:
            args = norm_test(loss,x,args,theta)
    return x, logger




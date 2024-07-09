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


def biased_quadratic(x,data = {}):
    assert 'A' in data.keys()
    assert 'b' in data.keys()
    if 'noise_A' in data.keys():
        assert 'random_state' in data.keys()
        A = data['A'] +\
        data['noise_A']*jnp.asarray(data['random_state'].randn(*data['noise_A'].shape))
    else:
        A = data['A']
    if 'noise_b' in data.keys():
        assert 'random_state' in data.keys()
        b = data['b'] +\
        data['noise_b']*jnp.asarray(data['random_state'].randn(*data['noise_b'].shape))
    else:
        b = data['b']
    res = jnp.einsum('ij,j->i',A,x)-b
    return jnp.linalg.norm(res)**2

def quadratic(x,data = {}):
    '''
    Returns jax function for (stochastic) quadratic

    '''
    assert 'AA' in data.keys()
    assert 'bA' in data.keys()
    assert 'bb' in data.keys()
    if 'noise_AA' in data.keys():
        assert 'random_state' in data.keys()
        my_noise = data['random_state'].randn(*data['AA'].shape)
        my_noise = my_noise@my_noise.T
        # Bias correction: this makes the Xi-square random matrix mean zero.
        my_noise -= (1./my_noise.shape[0])*np.eye(my_noise.shape[0])
        AA = data['AA'] + data['noise_AA']*jnp.asarray(my_noise)
    else:
        AA = data['AA']
    if 'noise_bA' in data.keys():
        assert 'random_state' in data.keys()
        bA = data['bA'] +\
        data['noise_bA']*jnp.asarray(data['random_state'].randn(*data['noise_bA'].shape))
    else:
        bA = data['bA']
    if 'noise_bb' in data.keys():
        assert 'random_state' in data.keys()
        bb = data['bb'] +\
        data['noise_bb']*jnp.asarray(data['random_state'].randn(*data['noise_bb'].shape))
    else:
        bb = data['bb']

    xAAx = jnp.einsum('i,i',x,jnp.einsum('ij,j->i',AA,x))   
    bAx = jnp.einsum('i,i',bA,x)
    return xAAx - 2*bAx + bb  


def setup_terms(dW = 100,damp = 1e-4):
    Q, _ = np.linalg.qr(np.random.randn(dW,dW))
    d = np.array([(0.1*jj)**1.5 for jj in range(1,dW+1)])
    A = (Q*d)@Q.T + damp*np.eye(dW)
    b = np.random.randn(dW)
    return A, b


def subsampled_quadratic(x,data = {}):
    """
    Stochastic approximation of \|Ax-b\|_2^2
    """

    assert 'A' in data.keys()
    assert 'b' in data.keys()
    assert 'p_A' in data.keys()
    p_A = data['p_A']
    assert 0 <= p_A < 1.0
    assert 'p_b' in data.keys()
    p_b = data['p_b']
    assert 0 <= p_b < 1.0
    m,d = data['A'].shape
    if p_A == 0.0:
        A = data['A']
    else:
        assert 'random_state' in data.keys()
        A = np.zeros_like(data['A'])
        A[:] = data['A']
        indices_A = data['random_state'].choice(np.arange(A.size),replace = False,\
                                                size = int(A.size*p_A))
        A[np.unravel_index(indices_A,A.shape)] = 0.0
    if p_b == 0.0:
        b = data['b']
    else:
        assert 'random_state' in data.keys()
        b = np.zeros_like(data['b'])
        b[:] = data['b']
        indices_b = data['random_state'].choice(np.arange(b.size),replace = False,\
                                                size = int(b.size*p_b))
        b[indices_b] = 0.0

    res = jnp.einsum('ij,j->i',A,x)-b
    return jnp.linalg.norm(res)**2


def default_true_data(loss,Aj,bj):
    if loss is biased_quadratic:
        data = {'A':Aj,'b':bj}
    elif loss is subsampled_quadratic:
        data = {'A':Aj,'b':bj, 'p_A':0.0, 'p_b':0.0}
    elif loss is quadratic:
        AAj = Aj.T@Aj
        bAj = bj.T@Aj
        bbj = bj.T@bj
        data = {'AA':AAj,'bA':bAj, 'bb':bbj}
    return data

def default_stochastic_data(loss,Aj,bj):
    if loss is biased_quadratic:
        stochastic_data = {'A':Aj, 'b':bj,\
                       'noise_A':0.01*np.linalg.norm(A),\
                       'noise_b':0.01*np.linalg.norm(b)}
    if loss is subsampled_quadratic:
        stochastic_data = {'A':Aj, 'b':bj,'p_A':0.5,'p_b':0.5}
    elif loss is quadratic:
        AAj = Aj.T@Aj
        bAj = bj.T@Aj
        bbj = bj.T@bj
        stochastic_data = {'AA':AAj, 'bA':bAj, 'bb':bbj,\
                       'noise_AA':0.1*jnp.linalg.norm(AAj),\
                       'noise_bA':0.1*jnp.linalg.norm(bAj),
                        'noise_bb':0.1*jnp.linalg.norm(bbj)}
    return stochastic_data

def default_true_data_from_stochastic_data(loss,stochastic_data):
    if loss is biased_quadratic:
        data = {'A':stochastic_data['A'],'b':stochastic_data['b']}
    elif loss is subsampled_quadratic:
        data = {'A':stochastic_data['A'],'b':stochastic_data['b'], 'p_A':0.0, 'p_b':0.0}
    elif loss is quadratic:
        data = {'AA':stochastic_data['AA'],'bA':stochastic_data['bA'],\
                 'bb':stochastic_data['bb']}
    return data

def quadratic_norm_test(loss,x,data,theta):
    true_data = default_true_data_from_stochastic_data(loss,data)
    g = jax.grad(loss)
    stoch_grad = g(x,data)
    true_grad = g(x,true_data)
    error = stoch_grad - true_grad
    error_norm = jnp.linalg.norm(error)
    tg_norm = jnp.linalg.norm(true_grad)
    if error_norm > theta*tg_norm:
        print('Bloody hell!')
        if loss == subsampled_quadratic:
                data['p_A'] = 0.5*data['p_A']
                data['p_b'] = 0.5*data['p_b']
                print(data['p_A'])
        elif loss ==  quadratic:
            data['noise_AA'] = 0.5*data['noise_AA']
            data['noise_bA'] = 0.5*data['noise_bA']
            data['noise_bb'] = 0.5*data['noise_bb']
        elif loss ==  biased_quadratic:
            data['noise_A'] = 0.5*data['noise_A']
            data['noise_b'] = 0.5*data['noise_b']
    return data

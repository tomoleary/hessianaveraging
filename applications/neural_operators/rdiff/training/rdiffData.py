# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

## Standard libraries
import os
import numpy as np

## JAX
import jax
import jax.numpy as jnp
from jax import random

## PyTorch
import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split


def get_rdiff_data(data_path, batch_size = 32, test_fraction = 0.1, input_projector = None,input_basis = None,\
						derivatives = False):

	mu_data = np.load(data_path+'/mq_data.npz')
	m_data = mu_data["m_data"]
	u_data = mu_data["q_data"]

	if derivatives:
		J_data = np.load(data_path+'/JTPhi_data.npz')['JTPhi_data'].transpose(0,2,1)

	dM = m_data.shape[1]
	dU = u_data.shape[1]

	if input_projector is not None:
		assert input_projector.shape[0] == dM
		m_data = np.einsum('mr,dm->dr',input_projector,m_data)
		dM = m_data.shape[1]

		if derivatives:
			J_data = np.einsum('mr,dqm->dqr',input_basis,J_data)

	ntotal = m_data.shape[0]
	ntest = int(test_fraction * ntotal)

	if derivatives:
		m_train, m_test, u_train, u_test, J_train, J_test = train_test_split(m_data, u_data, J_data, test_size=ntest)
	else:
		m_train, m_test, u_train, u_test = train_test_split(m_data, u_data, test_size=ntest)

	
	m_train = np.array(m_train, dtype = jnp.float32)
	u_train = np.array(u_train, dtype = jnp.float32)
	m_test = np.array(m_test, dtype = jnp.float32)
	u_test = np.array(u_test, dtype = jnp.float32)

	train_data = {'m': m_train, 'u': u_train}
	test_data = {'m':m_test, 'u':u_test}

	if derivatives:
		J_train = np.array(J_train, dtype = jnp.float32)
		J_test = np.array(J_test, dtype = jnp.float32)
		train_data['J'] = J_train
		test_data['J'] = J_test

	return train_data, test_data, dM, dU



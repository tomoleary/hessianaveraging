# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import numpy as np 
import scipy

import sys, os

files = os.listdir('.')
mat_files = []
for file in files:
	if file.endswith('.mat'):
		mat_files.append(file)

def isarray(item):
	return type(item) is np.ndarray or type(item) is scipy.sparse._csc.csc_matrix

for mat_file in mat_files:
	print('mat file = ',mat_file)
	mat_dict = scipy.io.loadmat(mat_file)
	print('key = ',mat_dict.keys())
	out_dict = {}
	for key,item in mat_dict.items():
		if isarray(item):
			if type(item) is scipy.sparse._csc.csc_matrix:
				item = np.array(item.todense())
			print('key = ',key)
			print('type(item) = ',type(item))
			print('item.shape = ',item.shape)
			out_dict[key] = item
		np.savez(mat_file.split('.mat')[0]+'.npz',**out_dict)


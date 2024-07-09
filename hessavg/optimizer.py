# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

################################################################################

class Optimizer(object):

	def __init__(self,loss,lr_schedule = None):
		self.loss = loss

		self.lr_schedule = lr_schedule
	

	def update(self,params,batch, hess_batch = None, batch_stats = None):
		raise NotImplementedError("Child class should implement method update")
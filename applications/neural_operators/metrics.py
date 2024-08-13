# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import jax
import jax.numpy as jnp

def squared_l2_error(y_true, y_pred):
    return squared_l2_norm(y_true - y_pred)

def squared_l2_norm(y):
    return jnp.inner(y, y)


def squared_f_error(y_true, y_pred):
    return squared_f_norm(y_true - y_pred)

def squared_f_norm(y):
    return jnp.sum(jnp.square(y))
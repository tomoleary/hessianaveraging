# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from typing import Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp

class GenericDense(nn.Module):
    layer_widths: Sequence[int]
    output_size: int 
    activation: str
    output_bias : bool = True

    def setup(self):
        assert self.activation in ["softplus", "tanh", "relu", "linear","gelu"]
        self.hidden_layers = [nn.Dense(width) for width in self.layer_widths]
        self.final_layer = nn.Dense(self.output_size, use_bias=self.output_bias)

    def __call__(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.activation == "softplus":
                x = nn.softplus(x)
            elif self.activation == "tanh":
                x = nn.tanh(x)
            elif self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "gelu":
                x = nn.gelu(x)
        x = self.final_layer(x)

        return x


def create_module_jacobian(module, mode="forward"):
    # @jax.jit
    def forward_pass(params, x):
        return module.apply(params, jnp.expand_dims(x, 0))[0]
    
    if mode == "forward":
        jac = jax.jacfwd(forward_pass, argnums=1)
    elif mode == "reverse":
        jac = jax.jacrev(forward_pass, argnums=1)
    else:
        raise ValueError("Incorrect AD mode")

    return jax.jit(jax.vmap(jac, (None, 0)))

class DINO(nn.Module):


    def __init__(self, network):
        super(DINO, self).__init__()
        self.network = network

        self.network_jacobian = create_module_jacobian(network)

    def init(self,*args,**kwargs):
        return self.network.init(*args,**kwargs)

    def apply_fn(self,params,x, **kwargs):
        function_value = self.network.apply(params,x)
        jacobian_value = self.network_jacobian(params,x)
        return function_value, jacobian_value




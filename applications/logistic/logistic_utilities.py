import jax
import jax.numpy as jnp 
from flax import linen as nn

class LogisticRegressionModel(nn.Module):
	def setup(self):
		self.layer = nn.Dense(1,use_bias = False)

	@nn.compact
	def __call__(self, x):
		return jnp.squeeze(self.layer(x))
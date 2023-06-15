import jax.numpy as jnp
from jax import jit

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jit(slow_f)

import time
start = time.time()
fast_f(x)
print("Fast took ", time.time() - start)

start = time.time()
slow_f(x)
print("Slow took", time.time() - start)

# Gradient Computation
from jax import grad
def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = grad(abs_val)
print(abs_val_grad(1.0))   # prints 1.0
print(abs_val_grad(-1.0))  # prints -1.0 (abs_val is re-evaluated)
# What ist the gradient at 0?
print(abs_val_grad(0.0))

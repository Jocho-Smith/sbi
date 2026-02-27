import torch
from sbi.inference import NPSE
from sbi.utils import BoxUniform

def linear_gaussian(theta, shift, cov):
    return theta + torch.randn_like(theta) #@ torch.linalg.cholesky(cov) + shift

num_dim = 1
likelihood_shift = -1.0 * torch.ones(num_dim)
likelihood_cov = 0.3 * torch.eye(num_dim)

# Uniform prior
prior = BoxUniform(-2 * torch.ones(num_dim), 2 * torch.ones(num_dim))

# NPSE inference with VP SDE
inference = NPSE(prior, sde_type="vp")

# Tiny dataset for fast reproduction
theta = prior.sample((5,))
x = linear_gaussian(theta, likelihood_shift, likelihood_cov)
inference.append_simulations(theta, x).train(stop_after_epochs=50)

# Build posterior (do not pass prior)
posterior = inference.build_posterior()

# Set default context x
posterior.set_default_x(torch.zeros(num_dim))

num_samples_test_param, steps_test_param = 1,1#10,20#50,1 #10,5

# Tiny sample request triggers the bug
num_samples = num_samples_test_param
samples = posterior.sample(sample_shape=(num_samples,), steps=steps_test_param)

print('num_samples: ', num_samples_test_param)
print('steps: ', steps_test_param)


print("Requested samples:", num_samples)
print("Returned samples shape:", samples.shape)
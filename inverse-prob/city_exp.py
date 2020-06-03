import numpy as onp
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

plt.style.use('seaborn')

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
NUM_WARMUP, NUM_SAMPLES = 1000, 20000


class Laplace(dist.Distribution):
    arg_constraints = {'loc': dist.constraints.real, 'scale': dist.constraints.positive}
    support = dist.constraints.real
    reparametrized_params = ['loc', 'scale']

    def __init__(self, loc=0., scale=1., validate_args=None):
        self.loc, self.scale = dist.util.promote_shapes(loc, scale)
        batch_shape = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        eps = random.laplace(key, shape=sample_shape + self.batch_shape + self.event_shape)
        return self.loc + eps * self.scale

    @dist.util.validate_sample
    def log_prob(self, value):
        normalize_term = np.log(1/(2*self.scale))
        value_scaled = np.abs(value - self.loc) / self.scale
        return -1*value_scaled + normalize_term

    @property
    def mean(self):
        return np.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self):
        return np.broadcast_to(2 * self.scale ** 2, self.batch_shape)


def forward(x1, x2, s):
    return s * np.sqrt(x1**2 + x2**2)


def model(obs):
    x1 = numpyro.sample('X1', Laplace(-1.5, .2))
    x2 = numpyro.sample('X2', Laplace(1, .2))
    # x1 = numpyro.sample('X1', dist.Uniform(-2, 2))
    # x2 = numpyro.sample('X2', dist.Uniform(-2, 2))
    s = numpyro.sample('S', dist.Normal(19.5, .5))
    t = forward(x1, x2, s)
    return numpyro.sample('obs', dist.Normal(t, 3/2), obs=obs)


if __name__ == '__main__':

    kernel = NUTS(model)
    mcmc = MCMC(kernel, NUM_WARMUP, NUM_SAMPLES)
    mcmc.run(rng_key_, obs=np.array([20]))
    mcmc.print_summary()

    samples = mcmc.get_samples()
    x1 = onp.array(samples['X1'])
    x2 = onp.array(samples['X2'])

    plt.figure()
    ax = plt.subplot()
    ax.plot(x1[::2], x2[::2], '.', alpha=.3, label='Possible locations')
    ax.set_aspect('equal')
    plt.title("City Map")
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    kde = gaussian_kde(np.vstack([x1, x2]))
    mode = minimize(lambda x: -kde(x), [-1, 0])
    x1_kp, x2_kp = mode.x

    ax.plot([x1_kp], [x2_kp], 'X', color='black', label='Kingpin')
    plt.legend(frameon=True, framealpha=1)

    P = onp.mean((x1 < -.5) & (x1 > -1.5) & (x2 < .5) & (x2 > -.5))
    print(f"Probability the kingpin is your neighbor: {100*P:.2f}%")

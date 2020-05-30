import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
NUM_WARMUP, NUM_SAMPLES = 1000, 20000


def gen_data():
    r = 1 + .1 * random.normal(rng_key, shape=(1000,))
    theta = np.linspace(-np.pi, np.pi, 1000)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    plt.figure()
    ax = plt.subplot()
    ax.plot(x, y, '.')
    ax.set_aspect('equal')
    plt.title("Data")
    plt.ylabel("y")
    plt.xlabel("x")

    plt.figure()
    ax = plt.subplot()
    r = forward(x, y)
    plt.hist(r)
    plt.title("output")
    plt.xlabel("r")

    return x, y


def forward(x, y):
    return np.sqrt(x**2 + y**2)


def model(r):
    # r ~ Normal(p, std)
    # std ~ Exp(1)
    # p ~ LogNormal(0, 5)

    # P(X, Y | R)

    # P(X)
    X = numpyro.sample('X', dist.Uniform(-10, 10))
    # P(Y)
    Y = numpyro.sample('Y', dist.Uniform(-10, 10))

    p = forward(X, Y)

    # P(std)
    sigma = numpyro.sample('sigma', dist.Exponential(1.))

    # P(R | X, Y, Std)
    if r is not None:
        return numpyro.sample('obs', dist.Normal(p, sigma), obs=r)
    # P(R)
    else:
        return numpyro.sample('obs', dist.Normal(p, sigma))


if __name__ == '__main__':

    X, Y = gen_data()
    r = forward(X, Y)

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, NUM_WARMUP, NUM_SAMPLES)
    mcmc.run(rng_key_, r=r)
    mcmc.print_summary()

    # P(X, Y | R)
    samples = mcmc.get_samples()

    plt.figure()
    ax = plt.subplot()
    ax.plot(samples['X'], samples['Y'], '.')
    ax.set_aspect('equal')
    plt.title("Guesses")
    plt.ylabel("y")
    plt.xlabel("x")

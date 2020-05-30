import matplotlib.pyplot as plt
import jax.numpy as np
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
NUM_WARMUP, NUM_SAMPLES = 1000, 20000

N_pts = 10


def gen_data():

    x_noise, y_noise = .2*random.normal(rng_key, (2, N_pts))
    x, y = forward(x_noise=x_noise, y_noise=y_noise)

    plt.figure()
    ax = plt.subplot()
    ax.plot(x, y, '.')
    ax.set_aspect('equal')
    plt.title("Data")
    plt.ylabel("y")
    plt.xlabel("x")

    return x, y


def forward(x=.2, y=0, width=1, height=.6, phi=3.14/5, x_noise=0, y_noise=0):
    t = np.linspace(0, 2*np.pi, N_pts)

    ellipse_x = x + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi) + x_noise/2.  # noqa: E501
    ellipse_y = y + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi) + y_noise/2.  # noqa: E501

    return [ellipse_x, ellipse_y]


def model(data):
    xc = numpyro.sample('xc', dist.Normal(0, 10))
    yc = numpyro.sample('yc', dist.Normal(0, 10))
    w = numpyro.sample('w', dist.LogNormal(0, 10))
    h = numpyro.sample('h', dist.LogNormal(0, 10))
    phi = numpyro.sample('phi', dist.Uniform(0, np.pi))
    px, py = forward(xc, yc, w, h, phi)
    sigma = numpyro.sample('sigma', dist.Exponential(1.))
    return numpyro.sample('obs', dist.Normal(np.vstack([px, py]), sigma), obs=data)


if __name__ == '__main__':

    X, Y = gen_data()

    # Run MCMC
    kernel = NUTS(model)
    mcmc = MCMC(kernel, NUM_WARMUP, NUM_SAMPLES)
    mcmc.run(rng_key_, np.vstack([X, Y]))
    mcmc.print_summary()
    samples = mcmc.get_samples()

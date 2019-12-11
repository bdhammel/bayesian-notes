import pyro
import torch
import torch.nn as nn
import numpy as np
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive, TracePredictive
from pyro.contrib.autoguide import AutoDiagonalNormal
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


batches = 10_000


def get_data():
    for _ in range(batches):
        x = np.random.uniform(low=-1, high=1, size=(512, 1))
        y = x**2
        yield torch.tensor(x.astype(np.float32)), torch.tensor(y.astype(np.float32))


torch_model = nn.Sequential(
    nn.Linear(1, 100),
    nn.Dropout(p=.5),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Dropout(p=.5),
    nn.Tanh(),
    nn.Linear(100, 1),
)


def model(x, y=None):
    priors = {}
    for name, par in torch_model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), .5*torch.ones(*par.shape)).independent(par.dim())

    bayesian_model = pyro.random_module('bayesian_model', torch_model, priors)
    sampled_model = bayesian_model()

    sigma = pyro.sample('sigma', dist.Uniform(0, 10))

    with pyro.plate('map', len(x)):
        prediction_mean = sampled_model(x).squeeze()
        pyro.sample('obs', dist.Normal(prediction_mean, sigma), obs=y)

    return prediction_mean


guide = AutoDiagonalNormal(model)


AdamArgs = {'lr': 1e-1, 'weight_decay': .1}
optimizer = torch.optim.Adam
sch = pyro.optim.ReduceLROnPlateau({'optimizer': optimizer, 'optim_args': AdamArgs,
                                    'verbose': True, 'patience': 19, 'min_lr': 1e-8})
svi = SVI(model, guide, sch, loss=Trace_ELBO())

losses = []
pbar = tqdm(get_data(), total=batches)
for i, (x, t) in enumerate(pbar):
    loss = svi.step(x, t)
    losses.append(loss)
    pbar.set_description(f"{loss:.5f}")


plt.figure()
plt.plot(losses)
plt.yscale('log')


def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }
    return site_stats


x, t = next(get_data())

predictive = Predictive(model, guide=guide, num_samples=80,
                        return_sites=("obs", "_RETURN"))
samples = predictive(x)
pred_summary = summary(samples)

mu = pred_summary["_RETURN"]
y = pred_summary["obs"]

predictions = pd.DataFrame({
    "x": x.detach().numpy().squeeze(),
    "mu_mean": mu["mean"].detach().numpy(),
    "mu_perc_5": mu["5%"].detach().numpy(),
    "mu_perc_95": mu["95%"].detach().numpy(),
    "y_mean": y["mean"].detach().numpy(),
    "y_perc_5": y["5%"].detach().numpy(),
    "y_perc_95": y["95%"].detach().numpy(),
    "true_y": t.detach().numpy().squeeze(),
})

plt.figure()
plt.plot(predictions.x, predictions.y_mean, '.')

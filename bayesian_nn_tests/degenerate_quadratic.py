import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, Adam
import numpy as np
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import Predictive
from pyro.contrib.autoguide import AutoDiagonalNormal
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


batches = 1_000

USE_GPU = torch.cuda.is_available()
if USE_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device('cuda' if USE_GPU else 'cpu')


def get_data():
    for _ in range(batches):
        x = np.random.uniform(low=-1, high=1, size=(20_000, 1))
        y = x**2
        yield torch.tensor(x.astype(np.float32)), torch.tensor(y.astype(np.float32))


def tonp(tensor):
    return tensor.detach().cpu().numpy()


x_test, t_test = next(get_data())

torch_model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1),
)
torch_model.to(device)

losses = []
optimizer = Adam(torch_model.parameters(), lr=1e-2)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
torch_model.train()
for _ in range(1):
    pbar = tqdm(get_data(), total=batches)

    for i, (x, t) in enumerate(pbar):
        optimizer.zero_grad()
        torch_model.zero_grad()
        preds = torch_model(x)
        loss = F.mse_loss(preds, t)
        loss.backward()

        losses.append(loss.item())
        optimizer.step()
        pbar.set_description(f"{losses[-1]:.5f}")

    scheduler.step(np.mean(losses[-500:]))

torch_model.eval()
preds = torch_model(x_test)

plt.figure()
plt.plot(losses)
plt.yscale('log')

plt.figure()
plt.plot(tonp(x_test), tonp(t_test), '.')
plt.plot(tonp(x_test), tonp(preds), '.')

plt.figure()
plt.plot(tonp(t_test) - tonp(preds), '.')


def model(x, y=None):
    priors = {}
    for name, par in torch_model.named_parameters():
        priors[name] = dist.Normal(torch.zeros(*par.shape), .5*torch.ones(*par.shape)).independent(par.dim())

    bayesian_model = pyro.random_module('bayesian_model', torch_model, priors)
    sampled_model = bayesian_model()

    sigma = pyro.sample('sigma', dist.Exponential(1))

    with pyro.plate('map', len(x)):
        prediction_mean = sampled_model(x).squeeze()
        pyro.sample('obs', dist.Normal(prediction_mean, sigma), obs=y)

    return prediction_mean


guide = AutoDiagonalNormal(model)

AdamArgs = {'lr': 1e-2}
optimizer = torch.optim.Adam
sch = pyro.optim.ReduceLROnPlateau({'optimizer': optimizer, 'optim_args': AdamArgs,
                                    'verbose': True, 'patience': 100, 'min_lr': 1e-8})
svi = SVI(model, guide, sch, loss=Trace_ELBO())

losses = []

pbar = tqdm(get_data(), total=batches)
for i, (x, t) in enumerate(pbar):
    loss = svi.step(x, t)
    losses.append(loss)
    pbar.set_description(f"{loss:.5f}")
    sch.step(np.mean(losses[-200:]))


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


predictive = Predictive(model, guide=guide, num_samples=80,
                        return_sites=("obs", "_RETURN"))
samples = predictive(x_test)
pred_summary = summary(samples)

mu = pred_summary["_RETURN"]
y = pred_summary["obs"]

predictions = pd.DataFrame({
    "x": tonp(x_test).squeeze(),
    "mu_mean": tonp(mu["mean"]),
    "mu_perc_5": tonp(mu["5%"]),
    "mu_perc_95": tonp(mu["95%"]),
    "y_mean": tonp(y["mean"]),
    "y_perc_5": tonp(y["5%"]),
    "y_perc_95": tonp(y["95%"]),
    "true_y": tonp(t_test).squeeze(),
})

plt.figure()
plt.plot(tonp(x_test), tonp(t_test), '.')
plt.plot(predictions.x, predictions.mu_mean, '.')
# plt.plot(predictions.x, predictions.y_mean, '.')

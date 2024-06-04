# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %matplotlib widget
# -

# +
from collections import defaultdict
from pprint import pprint

import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import is_interactive
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

from utils import extract_model_params, plot_samples, ExactGPModel


# +
torch.set_default_dtype(torch.float64)
torch.manual_seed(123)
# -

# # Generate toy 2D data


# +
class MexicanHat:
    def __init__(self, xlim, ylim, nx, ny, mode, **kwds):
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.xg, self.yg = self.get_xy_grid()
        self.XG, self.YG = torch.meshgrid(self.xg, self.yg, indexing="ij")
        self.X = self.make_X(mode)
        self.z = self.func(self.X)

    def make_X(self, mode="grid"):
        if mode == "grid":
            X = torch.empty((self.nx * self.ny, 2))
            X[:, 0] = self.XG.flatten()
            X[:, 1] = self.YG.flatten()
        elif mode == "rand":
            X = torch.rand(self.nx * self.ny, 2)
            X[:, 0] = X[:, 0] * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
            X[:, 1] = X[:, 1] * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return X

    def get_xy_grid(self):
        x = torch.linspace(self.xlim[0], self.xlim[1], self.nx)
        y = torch.linspace(self.ylim[0], self.ylim[1], self.ny)
        return x, y

    @staticmethod
    def func(X):
        r = torch.sqrt((X**2).sum(axis=1))
        return torch.sin(r) / r * 10

    @staticmethod
    def n2t(x):
        return torch.from_numpy(x)

    def apply_scalers(self, x_scaler, y_scaler):
        self.X = self.n2t(x_scaler.transform(self.X))
        Xg = x_scaler.transform(torch.stack((self.xg, self.yg), dim=1))
        self.xg = self.n2t(Xg[:, 0])
        self.yg = self.n2t(Xg[:, 1])
        self.XG, self.YG = torch.meshgrid(self.xg, self.yg, indexing="ij")
        self.z = self.n2t(y_scaler.transform(self.z[:, None])[:, 0])


# -

# +
data_train = MexicanHat(
    xlim=[-15, 5], ylim=[-15, 25], nx=20, ny=20, mode="rand"
)
x_scaler = StandardScaler().fit(data_train.X)
y_scaler = StandardScaler().fit(data_train.z[:, None])
data_train.apply_scalers(x_scaler, y_scaler)

data_pred = MexicanHat(
    xlim=[-15, 5], ylim=[-15, 25], nx=100, ny=100, mode="grid"
)
data_pred.apply_scalers(x_scaler, y_scaler)

# train inputs
X_train = data_train.X

# inputs for prediction and plotting
X_pred = data_pred.X

# noise-free train data
y_train = data_train.z

# noisy train data
##y_train = data_train.z + torch.randn(size=data_train.z.shape) / 5
# -

# +
# Cut out part of the train data to create out-of-distribution predictions
##mask = (X_train[:, 0] < -5) & (X_train[:, 1] < 0)
##X_train = X_train[~mask, :]
##y_train = y_train[~mask]
# -

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(
    data_pred.XG,
    data_pred.YG,
    data_pred.z.reshape((data_pred.nx, data_pred.ny)),
    color="tab:grey",
    alpha=0.5,
)
ax.scatter(
    xs=X_train[:, 0], ys=X_train[:, 1], zs=y_train, color="tab:blue", alpha=0.5
)
ax.set_xlabel("X_0")
ax.set_ylabel("X_1")

# # Define GP model

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)


# Inspect the model
print(model)

# Default start hyper params
pprint(extract_model_params(model, raw=False))

# +
# Set new start hyper params
model.mean_module.constant = 0.0
model.covar_module.base_kernel.lengthscale = 3.0
model.covar_module.outputscale = 8.0
model.likelihood.noise_covar.noise = 0.1

pprint(extract_model_params(model, raw=False))
# -

# # Fit GP to data: optimize hyper params

# +
# Train mode
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 300
history = defaultdict(list)
for ii in range(n_iter):
    optimizer.zero_grad()
    loss = -loss_func(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if (ii + 1) % 10 == 0:
        print(f"iter {ii+1}/{n_iter}, {loss=:.3f}")
    for p_name, p_val in extract_model_params(model).items():
        history[p_name].append(p_val)
    history["loss"].append(loss.item())
# -

# +
ncols = len(history)
fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 5, 5))
for ax, (p_name, p_lst) in zip(axs, history.items()):
    ax.plot(p_lst)
    ax.set_title(p_name)
    ax.set_xlabel("iterations")
# -

# +
# Values of optimized hyper params
pprint(extract_model_params(model, raw=False))
# -

# # Run prediction

# +
model.eval()
likelihood.eval()

with torch.no_grad():
    post_pred_f = model(X_pred)
    post_pred_y = likelihood(model(X_pred))

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
        data_pred.XG,
        data_pred.YG,
        data_pred.z.reshape((data_pred.nx, data_pred.ny)),
        color="tab:grey",
        alpha=0.5,
    )
    ax.plot_surface(
        data_pred.XG,
        data_pred.YG,
        post_pred_y.mean.reshape((data_pred.nx, data_pred.ny)),
        color="tab:red",
        alpha=0.5,
    )
    ax.set_xlabel("X_0")
    ax.set_ylabel("X_1")

assert (post_pred_f.mean == post_pred_y.mean).all()
# -


# # Plot difference to ground truth and uncertainty

# +
ncols = 3
fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 7, 5))

vmax = post_pred_y.stddev.max()

c0 = axs[0].contourf(
    data_pred.XG,
    data_pred.YG,
    torch.abs(post_pred_y.mean - data_pred.z).reshape(
        (data_pred.nx, data_pred.ny)
    ),
)
axs[0].set_title("|y_pred - y_true|")

c1 = axs[1].contourf(
    data_pred.XG,
    data_pred.YG,
    post_pred_f.stddev.reshape((data_pred.nx, data_pred.ny)),
    vmax=vmax,
)
axs[1].set_title("f_std (epistemic)")

c2 = axs[2].contourf(
    data_pred.XG,
    data_pred.YG,
    post_pred_y.stddev.reshape((data_pred.nx, data_pred.ny)),
    vmax=vmax,
)
axs[2].set_title("y_std (epistemic + aleatoric)")

for ax, c in zip(axs, [c0, c1, c2]):
    ax.set_xlabel("X_0")
    ax.set_ylabel("X_1")
    ax.scatter(x=X_train[:, 0], y=X_train[:, 1], color="white", alpha=0.2)
    fig.colorbar(c, ax=ax)


# When running as script
if not is_interactive():
    plt.show()
# -

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
# %matplotlib inline

import math
from collections import defaultdict
from pprint import pprint

import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import is_interactive


def extract_model_params(model, raw=False) -> dict:
    """Helper to convert model.named_parameters() to dict.

    With raw=True, use
        foo.bar.raw_param
    else
        foo.bar.param

    See https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Hyperparameters.html#Raw-vs-Actual-Parameters
    """
    if raw:
        return dict(
            (p_name, p_val.item())
            for p_name, p_val in model.named_parameters()
        )
    else:
        out = dict()
        # p_name = 'covar_module.base_kernel.raw_lengthscale'. Access
        # model.covar_module.base_kernel.lengthscale (w/o the raw_)
        for p_name, p_val in model.named_parameters():
            # Yes, eval() hack. Sorry.
            p_name = p_name.replace(".raw_", ".")
            p_val = eval(f"model.{p_name}")
            out[p_name] = p_val.item()
        return out


# Default float32 results in slightly noisy prior samples. Less so with
# float64. We get a warning with both
#   .../lib/python3.11/site-packages/linear_operator/utils/cholesky.py:40:
#       NumericalWarning: A not p.d., added jitter of 1.0e-08 to the diagonal
# but the noise is smaller w/ float64. Reason must be that the `sample()`
# method [1] calls `rsample()` [2] which performs a Cholesky decomposition of
# the covariance matrix. The default in
# np.random.default_rng().multivariate_normal() is method="svd", which is
# slower but seemingly a bit more stable.
#
# [1] https://docs.gpytorch.ai/en/stable/distributions.html#gpytorch.distributions.MultivariateNormal.sample
# [2] https://docs.gpytorch.ai/en/stable/distributions.html#gpytorch.distributions.MultivariateNormal.rsample
torch.set_default_dtype(torch.float64)

torch.manual_seed(123)
# -

# # Generate toy 2D data
#
# Here we generate noisy 1D data `X_train`, `y_train` as well as an extended
# x-axis `X_pred` which we use later for prediction also outside of the data
# range (extrapolation). The data has a constant offset `const` which we use to
# test learning a GP mean function $m(\ve x)$. We create a gap in the data to
# show how the model uncertainty will behave there.


# +
class SurfaceData:
    def __init__(self, xlim, ylim, nx, ny, mode, **kwds):
        self.xlim = xlim
        self.ylim = ylim
        self.nx = nx
        self.ny = ny
        self.xg, self.yg = self.get_xy_grid()
        self.XG, self.YG = torch.meshgrid(self.xg, self.yg, indexing="ij")
        self.X = self.make_X(mode)
        self.z = self.generate(self.X, **kwds)

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

    def func(self, X):
        raise NotImplementedError

    def generate(self, *args, **kwds):
        if "der" in kwds:
            der = kwds["der"]
            kwds.pop("der")
            if der == "x":
                return self.deriv_x(*args, **kwds)
            elif der == "y":
                return self.deriv_y(*args, **kwds)
            else:
                raise Exception("der != 'x' or 'y'")
        else:
            return self.func(*args, **kwds)


class MexicanHat(SurfaceData):
    def func(self, X):
        r = torch.sqrt((X**2).sum(axis=1))
        return torch.sin(r) / r

    def deriv_x(self, X):
        r = torch.sqrt((X**2).sum(axis=1))
        x = X[:, 0]
        return x * torch.cos(r) / r**2 - x * torch.sin(r) / r**3.0

    def deriv_y(self, X):
        r = torch.sqrt((X**2).sum(axis=1))
        y = X[:, 1]
        return y * torch.cos(r) / r**2 - y * torch.sin(r) / r**3.0


# -

# # Define GP model


# +
class ExactGPModel(gpytorch.models.ExactGP):
    """API:

    model.forward()             prior                   f_pred
    model()                     posterior               f_pred

    likelihood(model.forward()) prior with noise        y_pred
    likelihood(model())         posterior with noise    y_pred
    """

    def __init__(self, X_train, y_train, likelihood):
        super().__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# -

# +
data_train = MexicanHat(
    xlim=[-15, 5], ylim=[-15, 25], nx=20, ny=20, mode="rand"
)
data_pred = MexicanHat(
    xlim=[-15, 5], ylim=[-15, 25], nx=100, ny=100, mode="grid"
)


X_train = data_train.X
f_train = data_train.z
##y_train = f_train + torch.randn(size=f_train.shape) / 20
y_train = f_train

mask = (X_train[:, 0] < -5) & (X_train[:, 1] < 0)
# apply mask
X_train = X_train[~mask, :]
y_train = y_train[~mask]

X_pred = data_pred.X

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
# -

# +
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)
# -


# Inspect the model
print(model)

# Default start hyper params
pprint(extract_model_params(model, raw=False))

# +
# Set new start hyper params
model.mean_module.constant = 0.0
model.covar_module.base_kernel.lengthscale = 1.0
model.covar_module.outputscale = 1.0
model.likelihood.noise_covar.noise = 0.1

pprint(extract_model_params(model, raw=False))
# -


# +
# Train mode
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

n_iter = 200
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

ncols = len(history)
fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 5, 5))
for ax, (p_name, p_lst) in zip(axs, history.items()):
    ax.plot(p_lst)
    ax.set_title(p_name)
    ax.set_xlabel("iterations")
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

# # Plot difference to ground truth and uncertainty

# +
ncols = 2
fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 5, 5))

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
    post_pred_y.stddev.reshape((data_pred.nx, data_pred.ny)),
)
axs[1].set_title("y_std")

for ax, c in zip(axs, [c0, c1]):
    ax.set_xlabel("X_0")
    ax.set_ylabel("X_1")
    ax.scatter(x=X_train[:, 0], y=X_train[:, 1], color="white", alpha=0.2)
    fig.colorbar(c, ax=ax)
# -

# When running as script
if not is_interactive():
    plt.show()

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

# # Notation
# $\newcommand{\ve}[1]{\mathit{\boldsymbol{#1}}}$
# $\newcommand{\ma}[1]{\mathbf{#1}}$
# $\newcommand{\pred}[1]{\rm{#1}}$
# $\newcommand{\predve}[1]{\mathbf{#1}}$
# $\newcommand{\cov}{\mathrm{cov}}$
# $\newcommand{\test}[1]{#1_*}$
# $\newcommand{\testtest}[1]{#1_{**}}$
#
# Vector $\ve a\in\mathbb R^n$ or $\mathbb R^{n\times 1}$, so "column" vector.
# Matrix $\ma A\in\mathbb R^{n\times m}$. Design matrix with input vectors $\ve
# x_i\in\mathbb R^D$: $\ma X = [\ldots, \ve x_i, \ldots]^\top \in\mathbb
# R^{N\times D}$.
#
# We use 1D data, so in fact $\ma X \in\mathbb R^{N\times 1}$ is a vector, but
# we still denote the collection of all $\ve x_i = x_i\in\mathbb R$ points with
# $\ma X$ to keep the notation consistent with the slides.

# # Imports, helpers, setup

# ##%matplotlib notebook
# %matplotlib widget
# ##%matplotlib inline

# +
import math
from collections import defaultdict
from pprint import pprint

import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import is_interactive

from utils import extract_model_params, plot_samples


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

# # Generate toy 1D data
#
# Here we generate noisy 1D data `X_train`, `y_train` as well as an extended
# x-axis `X_pred` which we use later for prediction also outside of the data
# range (extrapolation). The data has a constant offset `const` which we use to
# test learning a GP mean function $m(\ve x)$. We create a gap in the data to
# show how the model uncertainty will behave there.


# +
def ground_truth(x, const):
    return torch.sin(x) * torch.exp(-0.2 * x) + const


def generate_data(x, gaps=[[1, 3]], const=5):
    y = ground_truth(x, const=const) + torch.randn(x.shape) * 0.1
    msk = torch.tensor([True] * len(x))
    if gaps is not None:
        for g in gaps:
            msk = msk & ~((x > g[0]) & (x < g[1]))
    return x[msk], y[msk], y


const = 5.0
x = torch.linspace(0, 4 * math.pi, 100)
X_train, y_train, y_gt_train = generate_data(x, gaps=[[6, 10]], const=const)
X_pred = torch.linspace(
    X_train[0] - 2, X_train[-1] + 2, 200, requires_grad=False
)
y_gt_pred = ground_truth(X_pred, const=const)

print(f"{X_train.shape=}")
print(f"{y_train.shape=}")
print(f"{X_pred.shape=}")

fig, ax = plt.subplots()
ax.scatter(X_train, y_train, marker="o", color="tab:blue", label="noisy data")
ax.plot(X_pred, y_gt_pred, ls="--", color="k", label="ground truth")
ax.legend()
# -

# # Define GP model
#
# We define the simplest possible textbook GP model using a Gaussian
# likelihood. The kernel is the squared exponential kernel with a scaling
# factor.
#
# $$\kappa(\ve x_i, \ve x_j) = s\,\exp\left(-\frac{\lVert\ve x_i - \ve x_j\rVert_2^2}{2\,\ell^2}\right)$$
#
# This makes two hyper params, namely the length scale $\ell$ and the scaling
# $s$. The latter is implemented by wrapping the `RBFKernel` with
# `ScaleKernel`.
#
# In addition, we define a constant mean via `ConstantMean`. Finally we have
# the likelihood noise $\sigma_n^2$. So in total we have 4 hyper params.
#
# * $\ell$ = `model.covar_module.base_kernel.lengthscale`
# * $\sigma_n^2$ = `model.likelihood.noise_covar.noise`
# * $s$ = `model.covar_module.outputscale`
# * $m(\ve x) = c$ = `model.mean_module.constant`


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
        """The prior, defined in terms of the mean and covariance function."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)
# -


# Inspect the model
print(model)

# Default start hyper params
pprint(extract_model_params(model, raw=False))

# +
# Set new start hyper params
model.mean_module.constant = 3.0
model.covar_module.base_kernel.lengthscale = 1.0
model.covar_module.outputscale = 1.0
model.likelihood.noise_covar.noise = 0.1

pprint(extract_model_params(model, raw=False))
# -


# # Sample from the GP prior
#
# We sample a number of functions $f_j, j=1,\ldots,M$ from the GP prior and
# evaluate them at all $\ma X$ = `X_pred` points, of which we have $N=200$. So
# we effectively generate samples from $p(\predve f|\ma X) = \mathcal N(\ve
# c, \ma K)$. Each sampled vector $\predve f\in\mathbb R^{N}$ and the
# covariance (kernel) matrix is $\ma K\in\mathbb R^{N\times N}$.

# +
model.eval()
likelihood.eval()

with torch.no_grad():
    # Prior
    M = 10
    pri_f = model.forward(X_pred)
    f_mean = pri_f.mean
    f_std = pri_f.stddev
    f_samples = pri_f.sample(sample_shape=torch.Size((M,)))
    print(f"{pri_f=}")
    print(f"{pri_f.mean.shape=}")
    print(f"{pri_f.covariance_matrix.shape=}")
    print(f"{f_samples.shape=}")
    fig, ax = plt.subplots()
    ax.plot(X_pred, f_mean, color="tab:red", label="mean", lw=2)
    plot_samples(ax, X_pred, f_samples, label="prior samples")
    ax.fill_between(
        X_pred,
        f_mean - 2 * f_std,
        f_mean + 2 * f_std,
        color="tab:orange",
        alpha=0.2,
        label="confidence",
    )
    ax.legend()
# -


# Let's investigate the samples more closely. A constant mean $\ve m(\ma X) =
# \ve c$ does *not* mean that each sampled vector $\predve f$'s mean is
# equal to $c$. Instead, we have that at each $\ve x_i$, the mean of
# *all* sampled functions is the same, so $\frac{1}{M}\sum_{j=1}^M f_j(\ve x_i)
# \approx c$ and for $M\rightarrow\infty$ it will be exactly $c$.
#

# Look at the first 20 x points from M=10 samples
print(f"{f_samples.shape=}")
print(f"{f_samples.mean(axis=0)[:20]=}")
print(f"{f_samples.mean(axis=0).mean()=}")
print(f"{f_samples.mean(axis=0).std()=}")

# Take more samples, the means should get closer to c
f_samples = pri_f.sample(sample_shape=torch.Size((M * 200,)))
print(f"{f_samples.shape=}")
print(f"{f_samples.mean(axis=0)[:20]=}")
print(f"{f_samples.mean(axis=0).mean()=}")
print(f"{f_samples.mean(axis=0).std()=}")

# # GP posterior predictive distribution with fixed hyper params
#
# Now we calculate the posterior predictive distribution $p(\test{\predve
# f}|\test{\ma X}, \ma X, \ve y)$, i.e. we condition on the train data (Bayesian
# inference).
#
# We use the fixed hyper param values defined above. In particular, since
# $\sigma_n^2$ = `model.likelihood.noise_covar.noise` is > 0, we have a
# regression setting.

# +
# Evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad():
    M = 10
    post_pred_f = model(X_pred)

    fig, ax = plt.subplots()
    f_mean = post_pred_f.mean
    f_samples = post_pred_f.sample(sample_shape=torch.Size((M,)))
    f_std = post_pred_f.stddev
    lower = f_mean - 2 * f_std
    upper = f_mean + 2 * f_std
    ax.plot(
        X_train.numpy(),
        y_train.numpy(),
        "o",
        label="data",
        color="tab:blue",
    )
    ax.plot(
        X_pred.numpy(),
        f_mean.numpy(),
        label="mean",
        color="tab:red",
        lw=2,
    )
    ax.plot(
        X_pred.numpy(),
        y_gt_pred.numpy(),
        label="ground truth",
        color="k",
        lw=2,
        ls="--",
    )
    ax.fill_between(
        X_pred.numpy(),
        lower.numpy(),
        upper.numpy(),
        label="confidence",
        color="tab:orange",
        alpha=0.3,
    )
    y_min = y_train.min()
    y_max = y_train.max()
    y_span = y_max - y_min
    ax.set_ylim([y_min - 0.3 * y_span, y_max + 0.3 * y_span])
    plot_samples(ax, X_pred, f_samples, label="posterior pred. samples")
    ax.legend()
# -


# # Fit GP to data: optimize hyper params
#
# In each step of the optimizer, we condition on the training data (e.g. do
# Bayesian inference) to calculate the posterior predictive distribution for
# the current values of the hyper params. We iterate until the log marginal
# likelihood is converged.
#
# We use a simplistic PyTorch-style hand written train loop without convergence
# control, so make sure to use enough `n_iter` and eyeball-check that the loss
# is converged :-)
#
# Observe how all hyper params converge. In particular, note that the constant
# mean $m(\ve x)=c$ converges to the `const` value in `generate_data()`.

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
        print(f"iter {ii + 1}/{n_iter}, {loss=:.3f}")
    for p_name, p_val in extract_model_params(model).items():
        history[p_name].append(p_val)
    history["loss"].append(loss.item())
# -

# Plot hyper params and loss (neg. log marginal likelihood) convergence
ncols = len(history)
fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(ncols * 5, 5))
for ax, (p_name, p_lst) in zip(axs, history.items()):
    ax.plot(p_lst)
    ax.set_title(p_name)
    ax.set_xlabel("iterations")

# Values of optimized hyper params
pprint(extract_model_params(model, raw=False))

# # Run prediction
#
# We show "noiseless" (left: $\sigma = \sqrt{\mathrm{diag}(\ma\Sigma)}$) vs.
# "noisy" (right: $\sigma = \sqrt{\mathrm{diag}(\ma\Sigma + \sigma_n^2\,\ma
# I_N)}$) predictions with
#
# $$\ma\Sigma = \testtest{\ma K} - \test{\ma K}\,(\ma K+\sigma_n^2\,\ma I)^{-1}\,\test{\ma K}^\top$$
#
# We find that $\ma\Sigma$ reflects behavior we would like to see from
# epistemic uncertainty -- it is high when we have no data
# (out-of-distribution). But this alone isn't the whole story. We need to add
# the estimated noise level $\sigma_n^2$ in order for the confidence band to
# cover the data.

# +
# Evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

with torch.no_grad():
    M = 10
    post_pred_f = model(X_pred)
    post_pred_y = likelihood(model(X_pred))

    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    fig_sigmas, ax_sigmas = plt.subplots()
    for ii, (ax, post_pred, name) in enumerate(
        zip(axs, [post_pred_f, post_pred_y], ["f", "y"])
    ):
        yf_mean = post_pred.mean
        yf_samples = post_pred.sample(sample_shape=torch.Size((M,)))

        yf_std = post_pred.stddev
        lower = yf_mean - 2 * yf_std
        upper = yf_mean + 2 * yf_std
        ax.plot(
            X_train.numpy(),
            y_train.numpy(),
            "o",
            label="data",
            color="tab:blue",
        )
        ax.plot(
            X_pred.numpy(),
            yf_mean.numpy(),
            label="mean",
            color="tab:red",
            lw=2,
        )
        ax.plot(
            X_pred.numpy(),
            y_gt_pred.numpy(),
            label="ground truth",
            color="k",
            lw=2,
            ls="--",
        )
        ax.fill_between(
            X_pred.numpy(),
            lower.numpy(),
            upper.numpy(),
            label="confidence",
            color="tab:orange",
            alpha=0.3,
        )
        if name == "f":
            sigma_label = r"$\pm 2\sqrt{\mathrm{diag}(\Sigma)}$"
            zorder = 1
        else:
            sigma_label = (
                r"$\pm 2\sqrt{\mathrm{diag}(\Sigma + \sigma_n^2\,I)}$"
            )
            zorder = 0
        ax_sigmas.fill_between(
            X_pred.numpy(),
            lower.numpy(),
            upper.numpy(),
            label="confidence " + sigma_label,
            color="tab:orange" if name == "f" else "tab:blue",
            alpha=0.5,
            zorder=zorder,
        )
        y_min = y_train.min()
        y_max = y_train.max()
        y_span = y_max - y_min
        ax.set_ylim([y_min - 0.3 * y_span, y_max + 0.3 * y_span])
        plot_samples(ax, X_pred, yf_samples, label="posterior pred. samples")
        if ii == 1:
            ax.legend()
    ax_sigmas.legend()
# -

# +
# When running as script
if not is_interactive():
    plt.show()
# -

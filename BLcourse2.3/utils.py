import gpytorch


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


def plot_samples(ax, X_pred, samples, label=None, **kwds):
    plot_kwds = dict(color="tab:green", alpha=0.3)
    plot_kwds.update(kwds)

    if label is None:
        ax.plot(X_pred, samples.T, **plot_kwds)
    else:
        ax.plot(X_pred, samples[0, :], **plot_kwds, label=label)
        ax.plot(X_pred, samples[1:, :].T, **plot_kwds, label="_")


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

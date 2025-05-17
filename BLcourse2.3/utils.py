from matplotlib import pyplot as plt
import torch


def extract_model_params(model, raw=False, try_item=False) -> dict:
    """Helper to convert model.named_parameters() to dict.

    With raw=True, use
        foo.bar.raw_param
    else
        foo.bar.param

    See https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Hyperparameters.html#Raw-vs-Actual-Parameters
    """
    if try_item:
        def get_val(p):
            if isinstance(p, torch.Tensor):
                if p.ndim == 0:
                    return p.item()
                else:
                    p_sq = p.squeeze()
                    if (p_sq.ndim == 1 and len(p_sq) == 1) or p_sq.ndim == 0:
                        return p_sq.item()
                    else:
                        return p
            else:
                return p
    else:
        get_val = lambda p: p
    if raw:
        return dict(
            (p_name, get_val(p_val))
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
            out[p_name] = get_val(p_val)
        return out


def fig_ax_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def plot_samples(ax, X_pred, samples, label=None, **kwds):
    plot_kwds = dict(color="tab:green", alpha=0.3)
    plot_kwds.update(kwds)

    if label is None:
        ax.plot(X_pred, samples.T, **plot_kwds)
    else:
        ax.plot(X_pred, samples[0, :], **plot_kwds, label=label)
        ax.plot(X_pred, samples[1:, :].T, **plot_kwds, label="_")

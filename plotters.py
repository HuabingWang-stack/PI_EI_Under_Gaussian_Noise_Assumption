import matplotlib.pyplot as plt
from bo_acquis import *

def plot_iteration(first_param_grid, sampled_params, sampled_loss, first_iter=0, alpha=1e-5,
                   greater_is_better=False, true_y=None, second_param_grid=None,
                   param_dims_to_plot=[0, 1], filepath=None, optimum=None,acquisition_func = expected_improvement):
    """ plot_iteration
    Plots a line plot (1D) or heatmap (2D) of the estimated loss function and
     acquisition function for each iteration of the Bayesian search algorithm.
    Arguments:
    ----------
        lock: Threading Lock
            prevent matplotlib instance rewritting in multithreading execution
        first_param_grid: array-like, shape = [n, 1]
            Array containing the grid of points to plot for the first parameter.
        sampled_params: array-like, shape = [n_points, n_params]
            Points for which the value of the loss function is computed.
        sampled_loss: function.
            Values of the loss function for the parameters in `sampled_params`.
        first_iter: int.
            Only plot iterations after the `first_iter`-th iteration.
        alpha: float
            Variance of the error term in the GP model.
        greater_is_better: boolean
            Boolean indicating whether we want to maximise or minimise the loss function.
        true_y: array-like, shape = [n, 1] or None
            Array containing the true value of the loss function. If None, the real loss
            is not plotted. (1-dimensional case)
        second_param_grid: array-like, shape = [n, 1]
            Array containing the grid of points to plot for the second parameter, in case
            of a heatmap.
        param_dims_to_plot: list of length 2
            List containing the indices of `sampled_params` that contain the first and
            second parameter.
        optimum: array-like [1, n_params].
            Maximum value of the loss function.
    """

    # Create the GP
    kernel = gp.kernels.Matern()
    model = gp.GaussianProcessRegressor(kernel=kernel,
                                        alpha=alpha,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)
    if acquisition_func in [probability_improvement, Mprobability_improvement]:
        title = 'Probability of Improvement'
    elif acquisition_func in [expected_improvement, Mexpected_improvement]:
        title = 'Expected Improvement'
    else:
        title = 'Acquisation Function'
    # Don't show the last iteration (next_sample is not available then)
    for i in range(first_iter, sampled_params.shape[0] - 1):
        model.fit(X=sampled_params[:(i + 1), :], y=sampled_loss[:(i + 1)])

        if second_param_grid is None:
            # 1-dimensional case: line plot
            mu, std = model.predict(first_param_grid[:, np.newaxis], return_std=True)
            if (acquisition_func == Mexpected_improvement) or (acquisition_func == Mprobability_improvement):
                acquisition_value = -1 * acquisition_func(first_param_grid, model, sampled_loss[:(i + 1)],sampled_params[:(i + 1), :],
                                           greater_is_better, 1)
            else:
                acquisition_value = -1 * acquisition_func(first_param_grid, model, sampled_loss[:(i + 1)],
                                            greater_is_better, 1)
            # lock.acquire()
            fig, ax1, ax2 = _plot_loss_1d(title,first_param_grid, sampled_params[:(i + 1), :], sampled_loss[:(i + 1)], mu, std, acquisition_value, sampled_params[i + 1, :], yerr=alpha, true_y=true_y)
        else:
            # Transform grids into vectors for acquisition value evaluation
            param_grid = np.array([[first_param, second_param] for first_param in first_param_grid for second_param in second_param_grid])

            mu, std = model.predict(param_grid, return_std=True)
            if (acquisition_func == Mexpected_improvement) or (acquisition_func == Mprobability_improvement):
                acquisition_value = -1 * acquisition_func(param_grid, model, sampled_loss[:(i + 1)],sampled_params[:(i + 1), :],
                                           greater_is_better, 2)
            else:
                acquisition_value = -1 * acquisition_func(param_grid, model, sampled_loss[:(i + 1)],
                                            greater_is_better, 2)
            # lock.acquire()
            fig, ax1, ax2 = _plot_loss_2d(title,first_param_grid, second_param_grid, sampled_params[:(i+1), param_dims_to_plot], sampled_loss, mu, acquisition_value, sampled_params[i + 1, param_dims_to_plot], optimum)

        if filepath is not None:
            plt.savefig('%s/bo_iteration_%d.png' % (filepath, i-1), bbox_inches='tight',dpi=400)
            plt.close()
        # lock.release()

def _plot_loss_1d(title, x_grid, x_eval, y_eval, mu, std, acquisition_value, next_sample, yerr=0.0, true_y=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True)

    # Loss function plot
    ax1.plot(x_grid, mu, label = "GP mean")
    ax1.fill_between(x_grid, mu - std, mu + std, alpha=0.5)
    ax1.errorbar(x_eval, y_eval, yerr, fmt='ok', zorder=3, label="Observed values")
    ax1.set_ylabel("Function value f(x)")
    ax1.set_xlabel("x")

    if true_y is not None:
        ax1.plot(x_grid, true_y, '--', label="True function")

    # Acquisition function plot
    ax2.plot(x_grid, acquisition_value, 'r', label=title)
    ax2.set_ylabel(title)
    ax2.set_title("Next sample point is C = %.3f" % next_sample)
    ax2.axvline(next_sample)

    return fig, ax1, ax2


def _plot_loss_2d(title,first_param_grid, second_param_grid, sampled_params, sampled_loss, mu, acquisition_value, next_sample, optimums=None):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8), sharex=True, sharey=True)

    X, Y = np.meshgrid(first_param_grid, second_param_grid, indexing='ij')

    # acquisition function contour plot
    cp = ax1.contourf(X, Y, acquisition_value.reshape(X.shape))
    plt.colorbar(cp, ax=ax1)
    ax1.set_title(title+". Next sample will be (%.2f, %.2f)" % (next_sample[0], next_sample[1]))
    ax1.autoscale(False)
    ax1.axvline(next_sample[0], color='k')
    ax1.axhline(next_sample[1], color='k')
    ax1.scatter(next_sample[0], next_sample[1])
    ax1.set_xlabel("x_1")
    ax1.set_ylabel("x_2")

    # Loss contour plot
    cp2 = ax2.contourf(X, Y, mu.reshape(X.shape))
    plt.colorbar(cp2, ax=ax2)
    ax2.autoscale(False)
    ax2.scatter(sampled_params[:, 0], sampled_params[:, 1], zorder=1)
    ax2.axvline(next_sample[0], color='k')
    ax2.axhline(next_sample[1], color='k')
    ax2.scatter(next_sample[0], next_sample[1])
    ax2.set_title("Mean estimate of loss surface for iteration %d" % (sampled_params.shape[0]-2))
    ax2.set_xlabel("x_1")
    ax2.set_ylabel("x_2")

    if optimums is not None:
        for optimum in optimums:
            ax2.scatter(optimum[0], optimum[1], marker='*', c='gold', s=150)

    return fig, ax1, ax2
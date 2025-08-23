from typing import List

from matplotlib.axes import Axes


def plot_training_hist(ax: Axes, Step: List[int], T_error: List[float], V_error: List[float]) -> Axes:
    """
    Plot training and validation error history during model training.

    Creates a log-log plot showing the evolution of training and validation errors
    over the course of training iterations.

    Parameters
    ----------
    ax
        Axes object for plotting the error history.
    Step
        List of training step numbers or iteration counts.
    T_error
        List of training error values corresponding to each step.
    V_error
        List of validation error values corresponding to each step.

    Returns
    -------
    Axes
        The modified axes object with the error plot.

    Notes
    -----
    The function creates a log-log scale plot with:
    - Training error shown as a solid red line
    - Validation error shown as a dashed blue line
    - Both axes using logarithmic scaling

    Examples
    --------
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> steps = [1, 10, 100, 1000, 10000]
    >>> train_errors = [1.0, 0.1, 0.01, 0.001, 0.0001]
    >>> val_errors = [1.2, 0.12, 0.012, 0.0012, 0.00012]
    >>> make_train_plot_hist(ax, steps, train_errors, val_errors)
    >>> plt.show()
    """
    ax.set_title("Error plot")
    ax.loglog(Step, T_error, "r", label="Training error")
    ax.loglog(Step, V_error, "b--", label="Validation error")
    ax.set_xlabel(r"Training times $n$")
    ax.set_ylabel(r"$l_1+l_2$")
    ax.legend()
    return ax

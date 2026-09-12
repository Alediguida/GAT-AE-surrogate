import os
from typing import Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics between ground truth and predicted values.

    Args:
        y_true (np.ndarray): 1D array of ground truth target values.
        y_pred (np.ndarray): 1D array of predicted target values.

    Returns:
        Dict[str, float]: Dictionary containing R2, MSE, RMSE, and MAE scores.
    """
    y_t = np.asarray(y_true).flatten()
    y_p = np.asarray(y_pred).flatten()

    r2 = r2_score(y_t, y_p)
    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)

    return {
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
    }


def plot_parity(
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_true: Optional[np.ndarray] = None,
    test_pred: Optional[np.ndarray] = None,
    title: str = "Latent Space Drag Prediction",
    xlabel: str = "True $C_d$ (CFD)",
    ylabel: str = "Predicted $C_d$ (Latent-MLP)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generates a parity plot (True vs Predicted values) with an ideal 1:1 fit line.

    Args:
        val_true (np.ndarray): Ground truth values for validation set.
        val_pred (np.ndarray): Predicted values for validation set.
        test_true (Optional[np.ndarray]): Ground truth values for test set.
        test_pred (Optional[np.ndarray]): Predicted values for test set.
        title (str): Plot title.
        xlabel (str): Label for the horizontal axis.
        ylabel (str): Label for the vertical axis.
        save_path (Optional[str]): Optional filepath to save the rendered figure.
        show (bool): Whether to display the plot with plt.show().

    Returns:
        Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects.
    """
    val_t = np.asarray(val_true).flatten()
    val_p = np.asarray(val_pred).flatten()
    r2_val = r2_score(val_t, val_p)

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(
        val_t,
        val_p,
        alpha=0.6,
        c="dodgerblue",
        s=60,
        edgecolors="w",
        linewidths=0.5,
        label=f"Validation ($R^2$ = {r2_val:.4f})",
    )

    all_trues = [val_t]
    all_preds = [val_p]

    if test_true is not None and test_pred is not None:
        test_t = np.asarray(test_true).flatten()
        test_p = np.asarray(test_pred).flatten()
        r2_test = r2_score(test_t, test_p)
        ax.scatter(
            test_t,
            test_p,
            alpha=0.9,
            c="darkorange",
            marker="^",
            s=80,
            edgecolors="k",
            linewidths=0.5,
            label=f"Test ($R^2$ = {r2_test:.4f})",
        )
        all_trues.append(test_t)
        all_preds.append(test_p)

    # Calculate axis bounds and ideal fit line
    combined_t = np.concatenate(all_trues)
    combined_p = np.concatenate(all_preds)
    min_val = min(combined_t.min(), combined_p.min())
    max_val = max(combined_t.max(), combined_p.max())
    margin = (max_val - min_val) * 0.05

    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        "k--",
        alpha=0.6,
        linewidth=2,
        label="Ideal Fit (y=x)",
    )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9, edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    return fig, ax


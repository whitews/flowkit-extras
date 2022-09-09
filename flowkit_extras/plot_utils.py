import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def plot_tsne_difference(
        tsne_results1,
        tsne_results2,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        fig_size=(16, 16)
):
    # fit an array of size [n_dim, n_samples]
    kde1 = gaussian_kde(
        np.vstack([tsne_results1[:, 0], tsne_results1[:, 1]])
    )
    kde2 = gaussian_kde(
        np.vstack([tsne_results2[:, 0], tsne_results2[:, 1]])
    )

    # evaluate on a regular grid
    x_grid = np.linspace(x_min, x_max, 250)
    y_grid = np.linspace(y_min, y_max, 250)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])

    z1 = kde1.evaluate(xy_grid)
    z2 = kde2.evaluate(xy_grid)

    z = z2 - z1

    # Plot the result as an image
    _, _ = plt.subplots(figsize=fig_size)
    plt.imshow(z.reshape(x_grid.shape),
               origin='lower', aspect='auto',
               extent=[x_min, x_max, y_min, y_max],
               cmap='bwr')
    plt.show()

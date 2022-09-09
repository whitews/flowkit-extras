import os
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from MulticoreTSNE import MulticoreTSNE
import seaborn
from matplotlib import cm
import matplotlib.pyplot as plt
from flowio.create_fcs import create_fcs


def calculate_extent(data_1d, d_min=None, d_max=None, pad=0.0):
    data_min = data_1d.min()
    data_max = data_1d.max()

    # determine padding to keep min/max events off the edge
    pad_d = max(abs(data_1d.min()), abs(data_1d.max())) * pad

    if d_min is None:
        d_min = data_min - pad_d
    if d_max is None:
        d_max = data_max + pad_d

    return d_min, d_max


def calculate_tsne(
        samples,
        n_dims=2,
        ignore_scatter=True,
        scale_scatter=True,
        transform=None,
        subsample=True
):
    """
    Performs dimensional reduction using the TSNE algorithm

    :param samples: List of FlowKit Sample instances on which to run TSNE
    :param n_dims: Number of dimensions to which the source data is reduced
    :param ignore_scatter: If True, the scatter channels are excluded
    :param scale_scatter: If True, the scatter channel data is scaled to be
      in the same range as the fluorescent channel data. If
      ignore_scatter is True, this option has no effect.
    :param transform: A Transform instance to apply to events
    :param subsample: Whether to sub-sample events from FCS files (default: True)

    :return: Dictionary of TSNE results where the keys are the FCS sample
      IDs and the values are the TSNE data for events with n_dims

    """
    tsne_events = None
    sample_events_lut = {}

    for s in samples:
        # Determine channels to include for TSNE analysis
        if ignore_scatter:
            tsne_indices = s.fluoro_indices
        else:
            # need to get all channel indices except time
            tsne_indices = list(range(len(samples[0].channels)))
            tsne_indices.remove(s.get_channel_index('Time'))

            # TODO: implement scale_scatter option
            if scale_scatter:
                pass

        s_events = s.get_raw_events(subsample=subsample)

        if transform is not None:
            fluoro_indices = s.fluoro_indices
            xform_events = transform.apply(s_events[:, fluoro_indices])
            s_events[:, fluoro_indices] = xform_events

        s_events = s_events[:, tsne_indices]

        # Concatenate events for all samples, keeping track of the indices
        # belonging to each sample
        if tsne_events is None:
            sample_events_lut[s.original_filename] = {
                'start': 0,
                'end': len(s_events),
                'channel_indices': tsne_indices,
                'events': s_events
            }
            tsne_events = s_events
        else:
            sample_events_lut[s.original_filename] = {
                'start': len(tsne_events),
                'end': len(tsne_events) + len(s_events),
                'channel_indices': tsne_indices,
                'events': s_events
            }
            tsne_events = np.vstack([tsne_events, s_events])

    # Scale data & run TSNE
    tsne_events = StandardScaler().fit(tsne_events).transform(tsne_events)
    tsne_results = MulticoreTSNE(n_components=n_dims, n_jobs=8).fit_transform(tsne_events)

    # Split TSNE results back into individual samples as a dictionary
    for k, v in sample_events_lut.items():
        v['tsne_results'] = tsne_results[v['start']:v['end'], :]

    # Return split results
    return sample_events_lut


def plot_tsne(
        session,
        tsne_results,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        fig_size=(8, 8)
):
    for s_id, s_results in tsne_results.items():
        sample = session.get_sample(s_id)
        tsne_events = s_results['tsne_results']

        for i, channel_idx in enumerate(s_results['channel_indices']):
            labels = sample.channels[str(channel_idx + 1)]

            x = tsne_events[:, 0]
            y = tsne_events[:, 1]

            # determine padding to keep min/max events off the edge,
            # but only if user didn't specify the limits
            x_min, x_max = calculate_extent(x, d_min=x_min, d_max=x_max, pad=0.02)
            y_min, y_max = calculate_extent(y, d_min=y_min, d_max=y_max, pad=0.02)

            z = s_results['events'][:, i]
            z_sort = np.argsort(z)
            z = z[z_sort]
            x = x[z_sort]
            y = y[z_sort]

            fig, ax = plt.subplots(figsize=fig_size)
            ax.set_title(" - ".join([s_id, labels['PnN'], labels['PnS']]))

            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

            seaborn.scatterplot(
                x,
                y,
                hue=z,
                palette=cm.get_cmap('rainbow'),
                legend=False,
                s=11,
                linewidth=0,
                alpha=0.7
            )

            file_name = s_id
            file_name = file_name.replace(".fcs", "")
            file_name = "_".join([file_name, labels['PnN'], labels['PnS']])
            file_name = file_name.replace("/", "_")
            file_name += ".png"
            plt.savefig(file_name)


def plot_tsne_difference(
        tsne_results1,
        tsne_results2,
        x_min=None,
        x_max=None,
        y_min=None,
        y_max=None,
        fig_size=(16, 16),
        export_fcs=False,
        export_cnt=20000,
        fcs_export_dir=None
):
    # fit an array of size [n_dim, n_samples]
    kde1 = gaussian_kde(
        np.vstack(
            [
                tsne_results1[:, 0],
                tsne_results1[:, 1]
            ]
        )
    )
    kde2 = gaussian_kde(
        np.vstack(
            [
                tsne_results2[:, 0],
                tsne_results2[:, 1]
            ]
        )
    )

    # evaluate on a regular grid
    x_grid = np.linspace(x_min, x_max, 250)
    y_grid = np.linspace(y_min, y_max, 250)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()])

    z1 = kde1.evaluate(xy_grid)
    z2 = kde2.evaluate(xy_grid)

    z = z2 - z1

    if export_fcs:
        z_g2 = z.copy()
        z_g2[z_g2 < 0] = 0
        z_g1 = z.copy()
        z_g1[z_g1 > 0] = 0
        z_g1 = np.abs(z_g1)

        z_g2_norm = [float(i) / sum(z_g2) for i in z_g2]
        z_g1_norm = [float(i) / sum(z_g1) for i in z_g1]

        cdf = np.cumsum(z_g2_norm)
        cdf = cdf / cdf[-1]
        values = np.random.rand(export_cnt)
        value_bins = np.searchsorted(cdf, values)
        new_g2_events = np.array([xy_grid[:, i] for i in value_bins])

        cdf = np.cumsum(z_g1_norm)
        cdf = cdf / cdf[-1]
        values = np.random.rand(export_cnt)
        value_bins = np.searchsorted(cdf, values)
        new_g1_events = np.array([xy_grid[:, i] for i in value_bins])

        pnn_labels = ['tsne_0', 'tsne_1']

        fh = open(os.path.join(fcs_export_dir, "tsne_group_1.fcs"), 'wb')
        create_fcs(new_g1_events.flatten(), pnn_labels, fh)
        fh.close()

        fh = open(os.path.join(fcs_export_dir, "tsne_group_2.fcs"), 'wb')
        create_fcs(new_g2_events.flatten(), pnn_labels, fh)
        fh.close()

    # Plot the result as an image
    _, _ = plt.subplots(figsize=fig_size)
    plt.imshow(z.reshape(x_grid.shape),
               origin='lower', aspect='auto',
               extent=[x_min, x_max, y_min, y_max],
               cmap='bwr')
    plt.show()

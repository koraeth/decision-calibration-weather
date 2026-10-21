"""Paper figures from metrics.nc (output of metrics.compute_metrics)."""
from itertools import product

import cartopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

PARAM_LABELS = {'threshold': r'$\theta$', 'cost_ratio': '$r$', 'shortfall_penalty': 'u-pen',
                'surplus_penalty': 'o-pen'}
VAR_LABELS = {'gap': 'cost gap', 'obs_cost': 'cost', 'exp_cost': 'exp cost'}
SKILL_TITLES = {'gap': 'Cost gap skill score', 'obs_cost': 'Cost skill score', 'exp_cost': 'Exp cost skill score'}
MAP_CBAR_LABELS = {'gap': 'Cost Gap Improvement', 'obs_cost': 'Cost Improvement', 'exp_cost': 'Exp Cost Improvement'}
PARAM_UNITS = {'threshold': ' °C'}

CMAP_COLORBLIND = LinearSegmentedColormap.from_list('colorblind_diverging', (   # as plotting.cmap_colorblind
    (0.000, (0.988, 0.016, 0.020)),
    (0.490, (1.000, 1.000, 1.000)),
    (1.000, (0.016, 0.000, 1.000))))


def relative_improvement(da, ref='ifs', percent=True):
    """(ref - ML) / |ref|: positive = ML lower (better). Drops the model dim. In % unless percent=False."""
    ml = da.drop_sel(model=ref).squeeze('model', drop=True)
    return (100 if percent else 1) * (da.sel(model=ref) - ml) / abs(da.sel(model=ref))


def _lead_axis(ax, lead, label_size=13):
    ax.set_xlabel('lead time', fontsize=label_size)
    ax.set_xticks(np.arange(len(lead)), lead)


def _model_label(m):
    return 'IFS' if m == 'ifs' else m


def _select(da, select):
    """Select task parameter values; scalars are kept as length-1 dims so they still appear in titles."""
    return da.sel({k: list(np.atleast_1d(v)) for k, v in (select or {}).items()})


def _panel_grid(sizes):
    """Rows = first parameter, cols = second; one parameter or a single column -> one row."""
    rows, cols = (1, sizes[0]) if len(sizes) == 1 else sizes
    return (1, rows) if cols == 1 else (rows, cols)


def plot_marginal(metrics, var='gap', mode='absolute', select=None, label_size=13, ylim=None,
                  legend=None, sharey=True):
    """var vs lead time, one panel per combination of the selected task parameters.

    mode: 'absolute' (both models) or 'relative' (improvement of ML over IFS in %).
    select: e.g. dict(threshold=[-4, 0, 4], cost_ratio=0.6); unselected parameters keep all values.
    legend: None (no legend) or ax.legend kwargs for the first panel (absolute mode),
            e.g. dict(loc='upper center', bbox_to_anchor=(0.5, 0.8)).
    """
    da = _select(metrics[var], select)
    params = [d for d in da.dims if d not in ('model', 'lead')]
    rows, cols = _panel_grid([da.sizes[p] for p in params])
    fig, axs = plt.subplots(rows, cols, figsize=(max(2, cols * 1.5), rows * 2.5),
                            squeeze=False, sharey=sharey, sharex=True)
    x = np.arange(da.sizes['lead'])

    for i, (ax, values) in enumerate(zip(axs.flat, product(*(da[p].values for p in params)))):
        panel = da.sel(dict(zip(params, values)))
        if mode == 'absolute':
            for m, marker, c in zip(panel.model.values, ['s', 'o'], ['C0', 'C1']):
                ax.plot(x, panel.sel(model=m), marker=marker, linestyle='-', c=c, label='IFS' if m == 'ifs' else m)
            ylabel = f'{VAR_LABELS[var]} (↓)'
        elif mode == 'relative':
            ax.plot(x, relative_improvement(panel), marker='o', linestyle='-', c='C0', label='rel. improvement')
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            ylabel = f'{VAR_LABELS[var]} impr. (%)'
        else:
            raise ValueError("mode must be 'absolute' or 'relative'")

        ax.set_title(', '.join(f'{PARAM_LABELS.get(p, p)}={v}' for p, v in zip(params, values)), fontsize=10)
        if i >= (rows - 1) * cols:                                    # bottom row
            ax.set_xlabel('lead time', fontsize=label_size)
            ax.set_xticks(x, da.lead.values)
        if i % cols == 0:                                             # first column
            ax.set_ylabel(ylabel, fontsize=label_size)
        if ylim is not None:
            ax.set_ylim(*ylim)

    if mode == 'absolute' and legend is not None:
        axs.flat[0].legend(**legend)
    fig.tight_layout()
    return fig


def plot_heatmap(metrics, var='gap', lead=15, select=None, cmap=CMAP_COLORBLIND, vmin=-0.2, vmax=0.2,
                 value_format='{:.2f}', figsize=(4.1, 2)):
    """Skill (IFS - ML) / |IFS| of var over the 2-D task-parameter grid at one lead time (fraction, not %).

    Rows = first task parameter (y axis), columns = second (x axis).
    """
    skill = relative_improvement(_select(metrics[var].sel(lead=lead), select), percent=False)
    row_param, col_param = skill.dims                    # e.g. threshold, cost_ratio
    data = skill.values

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    ax.set_xlabel(PARAM_LABELS.get(col_param, col_param))
    ax.set_ylabel(PARAM_LABELS.get(row_param, row_param))
    ax.set_xticks(np.arange(skill.sizes[col_param]), skill[col_param].values)
    ax.set_yticks(np.arange(skill.sizes[row_param]), skill[row_param].values)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, value_format.format(data[i, j]), ha='center', va='center',
                    color='white' if data[i, j] > 0.5 * data.max() else 'black')
    ax.set_title(f'{SKILL_TITLES[var]} (lead time {lead}d)')
    fig.tight_layout()
    return fig


def _diverging_cmap(vmin, vmax, vcenter=0):
    """Red -> white (at vcenter) -> blue, asymmetric if vcenter is off-centre (as plotting.create_asymm_cmap)."""
    mid = (vcenter - vmin) / (vmax - vmin)
    cdict = {'red':   ((0.0, 1.0, 1.0), (mid, 1.0, 1.0), (1.0, 0.0, 0.0)),
             'green': ((0.0, 0.0, 0.0), (mid, 1.0, 1.0), (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 0.0, 0.0), (mid, 1.0, 1.0), (1.0, 1.0, 1.0))}
    return LinearSegmentedColormap('custom_diverging', cdict, N=256)


def plot_map(metrics, var='gap', lead=1, select=None, vmin=-0.2, vmax=0.2, title=None, title_size=15, cbar=None,
             extent=None):
    """Skill (IFS - ML) / |IFS| of the time-mean var per grid point (fraction), for one task-parameter combination.

    select: one value per task parameter, e.g. dict(threshold=-4, cost_ratio=0.6); masked grid points stay blank.
    cbar: fig.colorbar kwargs, default dict(fraction=0.046, pad=0.04, label=MAP_CBAR_LABELS[var]).
    extent: (lon_min, lon_max, lat_min, lat_max) shown on the map; default: the full grid of metrics.
    """
    skill = relative_improvement(metrics[f'{var}_map'].sel(lead=lead, **(select or {})), percent=False)
    skill = skill.transpose('lat', 'lon')

    fig, ax = plt.subplots(figsize=(15, 3.5), dpi=100, subplot_kw={'projection': cartopy.crs.PlateCarree()})
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='black')
    ax.add_feature(cartopy.feature.BORDERS, edgecolor='black')
    cf = ax.pcolor(skill.lon, skill.lat, skill.values, shading='auto', transform=cartopy.crs.PlateCarree(),
                   cmap=_diverging_cmap(vmin, vmax), norm=Normalize(vmin=vmin, vmax=vmax))
    if extent is not None:
        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    if title is None:
        title = ', '.join(f'{PARAM_LABELS.get(p, p)} = {v}{PARAM_UNITS.get(p, "")}' for p, v in (select or {}).items())
    ax.set_title(title, fontsize=title_size)
    fig.colorbar(cf, ax=ax, **(cbar or dict(fraction=0.046, pad=0.04, label=MAP_CBAR_LABELS.get(var))))
    return fig


def plot_crps(metrics, ylim_rel=(-30, 30), label_size=13):
    """Left: mean CRPS of both models; right: relative improvement of ML over IFS (%)."""
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5), dpi=100)
    x = np.arange(metrics.sizes['lead'])
    for m, marker, c in zip(metrics.model.values, ['s', 'o'], ['C0', 'C1']):
        axs[0].plot(x, metrics.crps.sel(model=m), marker=marker, linestyle='-', c=c, label=_model_label(m))
    axs[0].set_title('CRPS (↓)')
    axs[1].plot(x, relative_improvement(metrics.crps), marker='o', linestyle='-', c='C0')
    axs[1].axhline(0, linestyle='--', color='k')
    axs[1].set_ylim(*ylim_rel)
    axs[1].set_title('CRPS impr. (%)')
    for ax in axs:
        _lead_axis(ax, metrics.lead.values, label_size)
    fig.tight_layout()
    return fig


def plot_pit(metrics):
    """PIT histograms (density) per lead time, both models; dashed line = uniform. From the stored counts."""
    edges = np.linspace(0, 1, metrics.sizes['pit_bin'] + 1)
    models = ['ifs'] + [m for m in metrics.model.values if m != 'ifs']          # IFS drawn below the ML model
    fig, axs = plt.subplots(1, metrics.sizes['lead'], figsize=(6, 1.4), dpi=100, sharey=True)
    for ax, lead in zip(axs, metrics.lead.values):
        for m, c, z in zip(models, ['C1', 'C0'], [-1, 1]):
            counts = metrics.pit_counts.sel(model=m, lead=lead).values
            ax.stairs(counts / (counts.sum() * np.diff(edges)), edges, color=c, linewidth=1.5, zorder=z,
                      label=_model_label(m), baseline=0)
        ax.axhline(1.0, c='black', linestyle='--', linewidth=1.5, alpha=0.8, zorder=-10)
        ax.set_title(f'lead time {lead} day' if lead == metrics.lead.values[0] else f'{lead} days')
    fig.tight_layout()
    return fig


def plot_ssr(metrics, ylim=(0.5, 1.1), label_size=13):
    """Spread/skill ratio vs lead time, both models; dashed line = 1."""
    fig, ax = plt.subplots(figsize=(1.75, 2.5), dpi=100)
    ax.axhline(1, linestyle='--', color='k', alpha=0.5)
    x = np.arange(metrics.sizes['lead'])
    for m, marker, c in zip(metrics.model.values, ['s', 'o'], ['C0', 'C1']):
        ax.plot(x, metrics.ssr.sel(model=m), marker=marker, linestyle='-', c=c, label=_model_label(m))
    ax.set_title('Spread/Skill ratio')
    _lead_axis(ax, metrics.lead.values, label_size)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig

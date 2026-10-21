"""Aggregate metrics from the per-pair scores (scores.nc): the numbers behind the figures."""
import numpy as np
import pandas as pd
import xarray as xr

COST_VARS = ['exp_cost', 'obs_cost', 'gap']
SPACE_TIME = ['lat', 'lon', 'time']
PIT_BINS = np.linspace(0, 1, 18)                   # 17 bins


def compute_metrics(scores, pit_bins=PIT_BINS):
    """C_exp, C_obs, C_gap (overall, per chosen action, as maps), CRPS, SSR, PIT counts.

    scores: per-pair Dataset ([model,] <task params>, lat, lon, lead, time), attribute n_actions; optional
    coordinate keep (lat, lon), True = evaluated. Means skip NaN: masked grid points and dates a model lacks.
    Shapes in the comments below: with a model dim; without one, it is simply absent.
    """
    actions = list(range(scores.attrs['n_actions']))
    ds = scores.where(scores.keep) if 'keep' in scores.coords else scores.copy()   # masked grid points -> NaN
    ds['gap'] = abs(ds.exp_cost - ds.obs_cost)
    costs = ds[COST_VARS]

    by_action = xr.concat([costs.where(ds.action == a).mean(SPACE_TIME) for a in actions],
                          dim=pd.Index(actions, name='action'))
    spread, rmse = np.sqrt(ds.ens_variance.mean('time')), np.sqrt(ds.sq_error.mean('time'))
    metrics = xr.merge([
        costs.mean(SPACE_TIME),                                                  # [model, params, lead]
        by_action.rename({v: f'{v}_by_action' for v in COST_VARS}),             # [model, params, action, lead]
        costs.mean('time').rename({v: f'{v}_map' for v in COST_VARS}),           # [model, params, lat, lon, lead]
        ds.crps.mean(SPACE_TIME).rename('crps'),                                 # [model, lead]
        ds.crps.mean('time').rename('crps_map'),                                 # [model, lat, lon, lead]
        (spread / rmse).mean(['lat', 'lon']).rename('ssr'),                      # [model, lead]
        _pit_counts(ds.pit, pit_bins),                                           # [model, lead, pit_bin]
    ])
    if 'keep' in scores.coords:
        metrics.coords['keep'] = scores.keep                                     # the evaluated grid points
    metrics.attrs.update(n_actions=len(actions))
    return metrics


def _pit_counts(pit, bins):
    """PIT histogram counts over all non-NaN pairs (lat, lon, time), per remaining dim (e.g. model, lead)."""
    def counts(v):
        return np.histogram(v[~np.isnan(v)], bins=bins)[0]
    hist = xr.apply_ufunc(counts, pit, input_core_dims=[SPACE_TIME], output_core_dims=[['pit_bin']], vectorize=True)
    return hist.assign_coords(pit_bin=(bins[:-1] + bins[1:]) / 2).rename('pit_counts')

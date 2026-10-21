"""Forecast scores per forecast-observation pair (DataArrays): CRPS, PIT, ensemble variance, squared error.

Spread and RMSE are aggregates: sqrt of the mean ensemble_variance / squared_error over the pairs.
compute_scores combines them with the decision costs into the per-pair Dataset of one forecast system (scores.nc).
"""
import xarray as xr
from properscoring import crps_ensemble

from decision_calibration.decisions import decision_costs_sweep


def compute_scores(task_cls, params, ens, obs, ens_task=None, obs_task=None, member_dim='member'):
    """Per-pair decisions, costs and forecast scores of one forecast system in one Dataset (scores.nc).

    task_cls, params: as in decision_costs_sweep (list values are swept).
    Forecast scores on ens / obs, decisions on the task variable ens_task / obs_task (default: ens / obs),
    e.g. wind speed vs. power. Several models: concatenate their results along a model dim, see README.
    """
    ens_task = ens if ens_task is None else ens_task
    obs_task = obs if obs_task is None else obs_task
    costs = decision_costs_sweep(task_cls, params, ens_task, obs_task, member_dim)
    forecast_scores = xr.Dataset(dict(crps=crps(ens, obs, member_dim), pit=pit(ens, obs, member_dim),
                                      ens_variance=ensemble_variance(ens, member_dim),
                                      sq_error=squared_error(ens, obs, member_dim)))
    ds = xr.merge([costs, forecast_scores])
    one_task = task_cls(**{k: v[0] if isinstance(v, list) else v for k, v in params.items()})
    ds.attrs.update(n_actions=one_task.n_actions)
    return ds


def crps(ens, obs, member_dim='member'):
    """Ensemble CRPS."""
    return xr.apply_ufunc(crps_ensemble, obs, ens, input_core_dims=[[], [member_dim]], join='exact')


def pit(ens, obs, member_dim='member'):
    """PIT: fraction of members <= the observation."""
    xr.align(ens, obs, join='exact')            # raise on mismatched coordinates instead of silently aligning
    return (ens <= obs).mean(member_dim)


def ensemble_variance(ens, member_dim='member'):
    """Variance over the members (ddof=1)."""
    return ens.var(member_dim, ddof=1)


def squared_error(ens, obs, member_dim='member'):
    """Squared error of the ensemble mean."""
    xr.align(ens, obs, join='exact')
    return (ens.mean(member_dim) - obs) ** 2

"""Bayes decisions from ensemble forecasts and their expected and observed costs."""
from functools import partial
from itertools import product

import numpy as np
import xarray as xr


def bayes_decision(task, ens, member_axis=-1):
    """Action with the lowest ensemble-mean cost and that expected cost.

    task: n_actions and cost(a, y), see decision_calibration.tasks.
    ens: ensemble forecast. Returns action index (ties: lowest index) and expected cost,
    both with the shape of ens without member_axis.
    """
    # members contiguous: the ensemble mean then uses pairwise summation, whatever the caller's memory layout
    # (strided input sums in another order -> last-bit differences, which can flip argmin on exact ties)
    ens = np.ascontiguousarray(np.moveaxis(np.asarray(ens), member_axis, -1))
    exp_cost_per_action = np.stack([task.cost(a, ens).mean(axis=-1) for a in range(task.n_actions)], axis=-1)
    action = exp_cost_per_action.argmin(axis=-1)
    exp_cost = np.take_along_axis(exp_cost_per_action, action[..., None], axis=-1)[..., 0]
    return action, exp_cost


def decision_costs(task, ens, obs, member_axis=-1):
    """Bayes decision from the ensemble, its expected cost C_exp and observed cost C_obs.

    obs: observation, shape of ens without member_axis. Returns a dict of arrays with that shape:
        action:   index of the Bayes action
        exp_cost: C_exp, its expected cost under the ensemble
        obs_cost: C_obs, its cost given the observation
    """
    action, exp_cost = bayes_decision(task, ens, member_axis)
    return dict(action=action, exp_cost=exp_cost, obs_cost=task.cost(action, obs))


def decision_costs_xr(task, ens, obs, member_dim='member'):
    """decision_costs for DataArrays: Dataset with action, exp_cost, obs_cost over the non-member dims.

    ens: DataArray with member_dim; obs: DataArray with the other dims of ens, any order.
    Coordinates of ens and obs must match exactly (no silent alignment).
    """
    # same steps as decision_costs; apply_ufunc moves member_dim last and matches the other dims by name
    action, exp_cost = xr.apply_ufunc(partial(bayes_decision, task), ens,
                                      input_core_dims=[[member_dim]], output_core_dims=[[], []])
    obs_cost = xr.apply_ufunc(task.cost, action, obs, join='exact')
    return xr.Dataset(dict(action=action, exp_cost=exp_cost, obs_cost=obs_cost))


def decision_costs_sweep(task_cls, params, ens, obs, member_dim='member'):
    """decision_costs_xr for every combination of task parameters, one dim per swept parameter.

    params: {name: value}; a list value is swept (becomes a dim, sorted ascending), anything else is fixed,
    e.g. dict(threshold=[-2, 0, 2], cost_ratio=[0.3, 0.7], transition_width=3).
    """
    swept = {k: v for k, v in params.items() if isinstance(v, list)}
    fixed = {k: v for k, v in params.items() if k not in swept}
    results = []
    for values in product(*swept.values()):
        kw = dict(zip(swept, values))
        ds = decision_costs_xr(task_cls(**fixed, **kw), ens, obs, member_dim)
        results.append(ds.expand_dims({k: [v] for k, v in kw.items()}))
    return xr.combine_by_coords(results)

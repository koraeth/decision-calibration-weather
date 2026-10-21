# Decision calibration for weather forecasts

Code for the paper

> Kornelius Raeth and Nicole Ludwig.
> **Forecast Skill is not Decision Skill: Evidence From Weather-Dependent Decision Tasks.**
> *Meteorological Applications* 33(5), e70260 (2026). https://doi.org/10.1002/met.70260

This repository provides the method as a small Python package (`decision_calibration`) for your own forecasts,
and the evaluation results and notebooks behind the paper figures (`paper/`).

## About

Weather forecasts are usually evaluated statistically, by comparing forecasts with
observations. However, in practice, forecasts are used to make decisions. Decision calibration
evaluates probabilistic forecasts from the decision-maker's perspective: at the level
of the decisions they inform rather than the forecasts themselves.

We use decision calibration to compare a machine learning model (ArchesWeatherGen) and
a numerical weather prediction model (ECMWF IFS ENS) on several weather-dependent
decision tasks (frost, heat, and wind power). The framework applies to any set of
forecast models.

Key findings:

- Performance at the forecast level does not reliably translate to performance in
  downstream decision-making.
- Some performance differences between models only become apparent at the decision level.
- Even among seemingly similar decision tasks, model rankings can change.

Typical forecast evaluations are therefore insufficient for selecting the best forecast
model for a specific decision task.

## What is measured

For each forecast–observation pair, the ensemble is used to choose the action with the lowest expected cost
(the Bayes action) for a decision task, i.e. a cost function of action and outcome. Then

- **C_exp**: the cost the forecast expects for that action (mean cost over the ensemble members),
- **C_obs**: the cost that actually occurs, given the observation,
- **C_gap**: the absolute difference |C_exp − C_obs| for that pair.

Averaged over all pairs, C_obs measures how good the decisions are, and C_gap how far the costs the forecast
anticipates are from the costs that occur; lower is better for both. C_gap is not zero even for a perfect
probabilistic forecast, because the outcome itself is uncertain; it is meant for comparing forecast models on
the same task. Three decision tasks are included (frost protection, heat protection, wind power bidding), and
your own can be added (see below).

## Installation

```bash
pip install git+https://github.com/koraeth/decision-calibration-weather
```

## Decision tasks

A decision task is a set of actions, numbered 0, 1, …, and a cost function `cost(a, y)`: the cost of taking
action `a` when the outcome is `y`. The three tasks of the paper are included:

| Task | Outcome `y` | Actions | Parameters |
|---|---|---|---|
| `FrostTask` | 2 m temperature (°C) | 0: no protection, 1: protect | `threshold`, `cost_ratio`, `transition_width` |
| `HeatTask` | 2 m temperature (°C) | 0: no protection, 1: protect | `threshold`, `cost_ratio`, `transition_width` |
| `WindPowerTask` | normalised wind power | index of the bid in `bids` (default 0, 0.1, …, 1) | `shortfall_penalty`, `surplus_penalty`, `zero_bid_cost`, `bids` |

The cost functions are described in the paper and plotted in [`paper/cost_functions.ipynb`](paper/cost_functions.ipynb); the class docstrings
give the exact formulas. Other tasks can be defined the same way: any object with `n_actions` and `cost(a, y)`,
see [`tasks.py`](src/decision_calibration/tasks.py).

## Usage

Forecasts and observations as xarray DataArrays; the ensemble has an extra `member` dim.

1. **Per forecast–observation pair:** `compute_scores(task_cls, params, ens, obs)` creates the task for every
   combination of parameters (list values in `params` are swept). For each pair, it takes the action with the
   lowest expected cost under the ensemble (the Bayes action) and returns that action with its expected and
   observed cost (C_exp, C_obs), plus standard forecast scores. If decisions use a different variable than
   the forecast scores (e.g. wind power vs. wind speed), pass it as `ens_task` / `obs_task`.
2. **Aggregates:** `compute_metrics(scores)` computes the decision scores (C_obs, C_gap) and the forecast scores
   per task parameter and lead time: means over space and time, per chosen action, or over time only (maps).
   It expects the dims `lat`, `lon` and `time` (plus e.g. `lead`); an optional boolean `keep` coordinate
   (`lat`, `lon`) restricts the evaluated grid points.

```python
import decision_calibration as dc

scores = dc.compute_scores(dc.FrostTask, dict(threshold=0, cost_ratio=[0.3, 0.7]), ens, obs)
metrics = dc.compute_metrics(scores)
```

For several models, compute the scores per model and concatenate them along a `model` dim
(`xr.concat(..., join='outer')`). The figures in [`paper/`](paper/) are made from such metrics.

## Paper figures and data

[`paper/`](paper/) contains the aggregated results of the paper (`metrics/*.nc`) and the notebooks that recreate
its figures; its README lists the setup (tasks, regions, dates). The forecasts themselves are not included.

- Forecasts: ECMWF IFS ENS and ArchesWeatherGen
- Verification: each model is verified against its own initial-condition source, following
  [GenCast](https://doi.org/10.1038/s41586-024-08252-9): ERA5 for ArchesWeatherGen and
  IFS HRES t=0 analysis for IFS ENS
- Availability: IFS ENS, IFS HRES t=0 and ERA5 via WeatherBench 2; ArchesWeatherGen forecasts on request

## Citation

```bibtex
@article{raeth_forecast_skill_decision_skill,
  author  = {Raeth, Kornelius and Ludwig, Nicole},
  title   = {Forecast Skill is not Decision Skill: Evidence From Weather-Dependent Decision Tasks},
  journal = {Meteorological Applications},
  year    = {2026},
  volume  = {33},
  number  = {5},
  pages   = {e70260},
  doi     = {10.1002/met.70260}
}
```

## License

MIT, see [LICENSE](LICENSE).

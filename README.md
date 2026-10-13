# Decision calibration for weather forecasts

Code for the paper

> Kornelius Raeth and Nicole Ludwig.
> **Forecast Skill is not Decision Skill: Evidence From Weather-Dependent Decision Tasks.**
> *Meteorological Applications* (in press).

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

## Data

- Forecasts: ECMWF IFS ENS and ArchesWeatherGen
- Verification: each model is verified against its own initial-condition source, following
  [GenCast](https://doi.org/10.1038/s41586-024-08252-9): ERA5 for ArchesWeatherGen and
  IFS HRES t=0 analysis for IFS ENS

## Citation

```bibtex
@article{raeth_forecast_skill_decision_skill,
  author  = {Raeth, Kornelius and Ludwig, Nicole},
  title   = {Forecast Skill is not Decision Skill: Evidence From Weather-Dependent Decision Tasks},
  journal = {Meteorological Applications},
  year    = {in press}
}
```

## License

MIT, see [LICENSE](LICENSE).

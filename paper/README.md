# Paper figures

Notebooks that recreate the figures of the paper from the shipped evaluation results.
The forecast data themselves are not included.

| File | Content |
|---|---|
| `figures.ipynb` | Result figures (cost gap and cost per task parameter, maps, CRPS, PIT, spread-skill ratio) from `metrics/<task>.nc`; the stored outputs show the wind power task, set `task_name` for the others |
| `cost_functions.ipynb` | Cost functions of the three tasks |
| `metrics/{frost,heat,windpower}.nc` | Aggregated results per task (output of `decision_calibration.compute_metrics`) |
| `figures.py` | Plotting functions |
| `conversions.py` | 10 m wind speed → hub-height wind speed → normalised power, as used for the wind power task |

Run from this folder after `pip install -e "..[paper]"`. Figures are written to `figures/`.
On first use, cartopy downloads coastline data (Natural Earth), which needs an internet connection.

## Setup

Both models are 50-member ensembles on a 1.5° grid, initialised daily in 2021 (IFS ENS: 1 Jan – 16 Dec, 350 dates;
ArchesWeatherGen: 2 Jan – 16 Dec, 349 dates), lead times 1, 3, 7, 10 and 15 days. Each model is verified against
its own initial-condition source: ERA5 for ArchesWeatherGen, IFS HRES t=0 for IFS ENS.

| | Frost | Heat | Wind power |
|---|---|---|---|
| Variable | 2 m temperature | 2 m temperature | normalised power from the 10 m wind speed (`conversions.py`: power law to 120 m, α = 0.1; power curve with cut-in 3, rated 13, cut-off 23 m/s) |
| Initialisation | 00 UTC | 12 UTC | 00 UTC |
| Evaluated grid points | land points in central Europe, 42–60° N, 9° W–28.5° E (217) | land points in southern and central Europe, 33–57° N, 7.5° W–34.5° E (302) | nearest grid point of each offshore wind farm in production (30) |
| Task parameters | threshold −4, −2, 0, 2, 4 °C; cost ratio 0.2, 0.3, …, 0.8; transition width 3 °C | threshold 30, 32, 34, 36, 38 °C; cost ratio 0.2, 0.3, …, 0.8; transition width 3 °C | bids 0, 0.1, …, 1; shortfall penalty 2, 3, 4; no surplus penalty; bid 0 costs 0.01 |

Transition width (`transition_width`, the buffer in the paper): range above (frost) or below (heat) the threshold
over which the costs change linearly. Decisions for wind power use normalised power; CRPS, PIT and spread-skill ratio use the 120 m wind speed.
Land points: `Basemap.is_land` at the grid point. Wind farms: dataset on offshore wind farms provided by the European
Marine Observation and Data Network (EMODnet), farms with status "Production". The evaluated grid points of each task are stored as the `keep` coordinate in `metrics/<task>.nc`.

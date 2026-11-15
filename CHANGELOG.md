# Changelog

All notable changes to TSForecasting are documented in this file.

## [2.1.0]

A correctness and robustness release. The public API is unchanged; existing code
keeps running, but **evaluation metrics, model selection and prediction intervals
can differ from 2.0.0** because the evaluation no longer leaks future information.

### Fixed - results that were silently wrong

- **Look-ahead leakage in the expanding window evaluation.** A supervised row built
  at forecast origin `t` carries the targets `t+1 ... t+horizon`. The `horizon - 1`
  rows right before each test window held targets that were not yet observed at the
  window's forecast origin, and they were used for training. They are now purged, so
  each window is fitted only on information available at its origin - the same
  information the final `forecast()` call can use. Reported errors were optimistic
  (up to ~15% on trending and random-walk series) and could favour the wrong model;
  prediction intervals, which are calibrated on those residuals, were too narrow.
- **AutoGluon target leakage.** Each horizon's predictor was trained with the other
  horizons' targets as input features, and received them again when predicting. The
  predictors now only see the lag/date features and their own target.
- **`datetime_features=False` crashed** during fitting (the raw `Date` column was sent
  to the scaler) and was ignored by `forecast()`. Both paths now honour the setting.
- **Unsorted datasets** produced meaningless lags and forecast timestamps. Datasets are
  now sorted chronologically (with a warning).
- **`model_configurations()` returned the registry's own dictionaries.** Editing the
  result, as the documentation suggests, changed the defaults of every model created
  afterwards in the same session. A deep copy is returned.
- **Monthly forecast timestamps drifted** for month-end series (Jan 31 -> Feb 28 ->
  Mar 28 -> ...). The day-of-month convention of the series is now preserved.
- **Conformal intervals** used an interpolated percentile instead of the
  `ceil((n + 1) * coverage)`-th order statistic required by the coverage guarantee.
- **Interval estimators with 1-D inputs** treated `n` residuals of one horizon as one
  residual of `n` horizons, which produced degenerate or NaN bounds.

### Fixed - crashes and unclear errors

- One failing model (for example `GeneralizedLR` on series with negative values) no
  longer aborts the run: it is skipped with a `RuntimeWarning` and reported in
  `TSForecasting.failed_models`. `ModelFitError` is raised only if every model fails.
- Configurations with no valid evaluation window (typically `sliding_size < horizon`
  on small datasets) ended in `KeyError: 'Model'`; they now raise
  `InsufficientDataError` describing the configuration.
- Missing values, infinite values, non-numeric targets, `NaT` and duplicated
  timestamps are reported with a `DatasetError` instead of an error from deep inside
  scikit-learn or NumPy.
- Builder setters (`with_train_size`, `with_lags`, ...) accepted out-of-range values.
  Every setter is validated and raises `ConfigurationError`.
- Changing the configuration of a fitted pipeline left it in an inconsistent state; the
  pipeline now requires a new `fit_forecast()`.
- `TimeSeriesDatasetGenerator.quick_monthly/quick_daily/quick_hourly` failed with
  "got multiple values for keyword argument" when a default such as
  `seasonality_period` was overridden.
- `TimeSeriesDatasetGenerator.list_available_datasets()`, used by the examples, did
  not exist.

### Added

- `verbose` parameter (`TSForecasting(verbose=False)`, `.with_verbose(False)`) to
  silence the evaluation progress.
- Models registered with `ModelRegistry.register` can be used in the pipeline by name;
  `register` accepts `default_params`, and `ModelRegistry.unregister` /
  `ModelRegistry.resolve_name` were added.
- Model names are case-insensitive (`"xgboost"`, `"CatBoost"`); granularity, metric
  and scaler aliases (`"daily"`, `"mape"`, `"MinMax"`) are accepted by the pipeline.
- `TSForecasting.failed_models`, `TSForecasting.is_fitted`.
- `forecast()` caches the final model fit: calling it again on the same data, for
  example to compare interval methods, does not retrain the model.
- A warning when the configured `granularity` does not match the spacing of `Date`,
  and when columns other than `Date` and `y` are passed (they are not used).
- `ModelFitError`; `ConfigurationError`, `InsufficientDataError`, `ModelFitError`,
  `PerformanceHistory`, `IntervalEstimatorFactory` and `compute_intervals` are exported
  from the package root.
- `[dev]` extra (`pip install -e ".[dev]"`).

### Changed

- Input-related exceptions (`ConfigurationError`, `ModelNotFoundError`, `DatasetError`,
  `InsufficientDataError`) also derive from `ValueError`. Invalid constructor
  arguments raise `ConfigurationError` instead of a raw pydantic/enum error.
- `history()` returns copies of the internal DataFrames.
- Leaderboard: `Max Error` is the largest error observed (it was the mean of the
  per-window maxima, while the per-horizon table already used the maximum); the
  leaderboard index is reset so it reflects the ranking.
- `TimeSeriesDatasetGenerator` draws from a private random stream and no longer
  reseeds NumPy's global generator. Values for a given `random_state` are unchanged.
  `random_state=None` is supported.
- Weekly synthetic series start at `start_date` instead of the following Sunday.
- `GBR` no longer forwards the `criterion` argument unless set explicitly (deprecated
  in scikit-learn 1.9); pandas `dayofweek`/`dayofyear` deprecations were removed.
- Windowing in `Processing.make_timeseries` is vectorised (about 4-5x faster on 100k rows).
- The Gaussian interval z-score uses the standard library; SciPy is no longer imported.

### Documentation

- Pipeline diagram (`docs/TSForecasting_Pipeline_Diagram.drawio` and `.png`) redrawn for 2.1: purged
  expanding windows, per-window scaling, failure isolation, the `datetime_features` switch, interval
  calibration and the actual outputs. The README now shows it from the repository instead of an external
  image host, without a fixed height that distorted it.

### Packaging

- `pydantic>=2.0` is declared as a dependency (it was imported but not declared).
- `xgboost==2.0.3` relaxed to `xgboost>=2.0.3`.
- `requirements.txt` is shipped in the source distribution (installing from the sdist
  failed), the invalid `py_modules` entry was removed, `python_requires>=3.10` and a
  PEP 517 `[build-system]` table were added, and the version is read from
  `tsforecasting/__init__.py`.

## [2.0.0]

- Modular architecture (core, models, processing, evaluation, pipeline), builder
  pattern, prediction intervals and synthetic data generation.

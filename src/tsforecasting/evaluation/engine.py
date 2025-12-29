from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from tsforecasting.core.enums import Granularity, ScalerType
from tsforecasting.core.exceptions import InsufficientDataError
from tsforecasting.core.schemas import ForecastConfig, PreprocessingConfig
from tsforecasting.models.base import AutoGluonForecaster, BaseForecaster
from tsforecasting.processing.processor import Processing


class EvaluationEngine:
    """Handles expanding window evaluation logic.
    
    Manages the complete evaluation workflow including:
    - Time series transformation
    - Train/test splitting across windows
    - Feature scaling
    - Model fitting and prediction
    - Metric computation and aggregation
    """
    
    def __init__(
        self,
        config: ForecastConfig,
        preprocessing: PreprocessingConfig,
    ):
        """Initialize evaluation engine.
        
        Args:
            config: Forecasting configuration.
            preprocessing: Preprocessing configuration.
        """
        self._config = config
        self._preprocessing = preprocessing
        self._processor = Processing()
        self._results: List[pd.DataFrame] = []
        self._input_cols: List[str] = []
        self._target_cols: List[str] = []
        self._autogluon_fitted: Dict[str, bool] = {}
        self._autogluon_models: Dict[str, AutoGluonForecaster] = {}
    
    def _get_scaler(self):
        """Get scaler based on configuration."""
        scalers = {
            ScalerType.STANDARD: StandardScaler,
            ScalerType.MINMAX: MinMaxScaler,
            ScalerType.ROBUST: RobustScaler,
        }
        return scalers[self._preprocessing.scaler]()
    
    def evaluate(
        self,
        dataset: pd.DataFrame,
        models: Dict[str, BaseForecaster],
    ) -> pd.DataFrame:
        """Run expanding window evaluation for all models.
        
        Args:
            dataset: Input DataFrame with Date and y columns.
            models: Dictionary mapping model names to forecaster instances.
            
        Returns:
            DataFrame with all forecast results.
        """
        self._results = []
        
        # Transform to time series format
        timeseries = self._processor.make_timeseries(
            dataset=dataset,
            window_size=self._config.lags,
            horizon=self._config.horizon,
            datetime_engineering=self._preprocessing.datetime_features,
        )
        
        # Identify columns
        self._target_cols = [c for c in timeseries.columns if "horizon" in c]
        self._input_cols = [c for c in timeseries.columns if c not in self._target_cols]
        
        # Evaluate each model
        for model_name, model in models.items():
            model_results = self._evaluate_model(
                timeseries=timeseries,
                model_name=model_name,
                model=model,
            )
            self._results.extend(model_results)
        
        return pd.concat(self._results, ignore_index=True) if self._results else pd.DataFrame()
    
    def _evaluate_model(
        self,
        timeseries: pd.DataFrame,
        model_name: str,
        model: BaseForecaster,
    ) -> List[pd.DataFrame]:
        """Evaluate single model across all windows.
        
        Args:
            timeseries: Transformed time series DataFrame.
            model_name: Model identifier.
            model: Forecaster instance.
            
        Returns:
            List of DataFrames with results per window.
        """
        results = []
        
        # Prepare data (exclude last horizon rows)
        df = timeseries.iloc[:-self._config.horizon].copy()
        train_size = int(self._config.train_size * df.shape[0])
        iterations = int(df.iloc[train_size:].shape[0] / self._config.sliding_size)
        
        if iterations < 1:
            raise InsufficientDataError(
                f"Insufficient data for evaluation. "
                f"Need at least {self._config.sliding_size} samples after train split."
            )
                
        for rolling_cycle in range(iterations):
            current_train_size = train_size + (rolling_cycle * self._config.sliding_size)
            
            train = df.iloc[:current_train_size].copy()
            test = df.iloc[current_train_size:].copy()
            
            if test.shape[0] >= self._config.horizon:
                print(
                    f"Algorithm Evaluation: {model_name} || "
                    f"Window Iteration: {rolling_cycle + 1} of {iterations}"
                )
                print(f"Rows Train: {train.shape[0]}")
                print(f"Rows Test: {test.shape[0]}")
                
                if rolling_cycle + 1 == iterations:
                    print(" ")
                
                # Preserve dates before scaling
                test_dates = test.index if "Date" not in test.columns else test["Date"].values
                
                # Scale features
                scaler = self._get_scaler()
                scaler.fit(train[self._input_cols])
                train[self._input_cols] = scaler.transform(train[self._input_cols])
                test[self._input_cols] = scaler.transform(test[self._input_cols])
                
                # Fit and predict
                y_pred = self._fit_predict(
                    train=train,
                    test=test,
                    model_name=model_name,
                    model=model,
                )
                
                # Collect results
                y_trues = test[self._target_cols].iloc[:self._config.horizon]
                y_preds = y_pred.iloc[:self._config.horizon]
                
                result_df = pd.concat([y_trues.reset_index(drop=True), 
                                    y_preds.reset_index(drop=True)], axis=1)
                result_df.insert(0, "Date", test_dates[:self._config.horizon])
                result_df["Window"] = rolling_cycle + 1
                result_df["Model"] = model_name
                
                results.append(result_df)

        return results
    
    def _fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model_name: str,
        model: BaseForecaster,
    ) -> pd.DataFrame:
        """Fit model and generate predictions.
        
        Args:
            train: Training DataFrame.
            test: Test DataFrame.
            model_name: Model identifier.
            model: Forecaster instance.
            
        Returns:
            DataFrame with predictions.
        """
        X_train = train[self._input_cols].values
        X_test = test[self._input_cols].values
        y_train = train[self._target_cols].values
        
        # Handle AutoGluon specially
        if model_name == "AutoGluon":
            return self._fit_predict_autogluon(train, test, model)
        
        # Standard sklearn-style models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Create prediction DataFrame
        forecast_cols = [
            f"y_forecast_horizon_{i + 1}" 
            for i in range(y_pred.shape[1])
        ]
        
        return pd.DataFrame(y_pred, columns=forecast_cols, index=test.index)
    
    def _fit_predict_autogluon(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: AutoGluonForecaster,
    ) -> pd.DataFrame:
        """Handle AutoGluon fitting and prediction.
        
        AutoGluon is fitted once and reused for subsequent predictions.
        """
        model_key = "AutoGluon"
        
        if model_key not in self._autogluon_fitted or not self._autogluon_fitted[model_key]:
            print(" ")
            model.set_labels(self._target_cols)
            model.fit(train)
            self._autogluon_fitted[model_key] = True
            self._autogluon_models[model_key] = model
        else:
            model = self._autogluon_models[model_key]
        
        y_pred = model.predict(test)
        
        # Rename columns
        y_pred.columns = [
            f"y_forecast_horizon_{col.split('_')[-1]}" 
            if "horizon" in col else col 
            for col in y_pred.columns
        ]
        
        return y_pred
    
    def single_fit_predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        model: BaseForecaster,
        model_name: str = "model",
    ) -> pd.DataFrame:
        """
        Single fit-predict cycle for final forecasting.
        
        Args:
            train: Training DataFrame (already transformed).
            test: Test DataFrame (already transformed).
            model: Forecaster instance.
            model_name: Model identifier.
            
        Returns:
            DataFrame with predictions.
        """
        X_train = train[self._input_cols].values
        X_test = test[self._input_cols].values
        y_train = train[self._target_cols].values
        
        if model_name == "AutoGluon":
            if isinstance(model, AutoGluonForecaster):
                model.set_labels(self._target_cols)
                model.fit(train)
                return model.predict(test)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        forecast_cols = [
            f"y_forecast_horizon_{i + 1}" 
            for i in range(y_pred.shape[1])
        ]
        
        return pd.DataFrame(y_pred, columns=forecast_cols, index=test.index)
    
    @property
    def input_cols(self) -> List[str]:
        """Return input column names."""
        return self._input_cols
    
    @property
    def target_cols(self) -> List[str]:
        """Return target column names."""
        return self._target_cols
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to load inputs and train a model using Quantile Regression Random Forest
(QRF)."""

import pathlib
import warnings
from pathlib import Path
from typing import Optional, Union

import iris
import numpy as np
import pandas as pd

from improver import PostProcessingPlugin
from improver.calibration import (
    CalibrationSchemas,
    add_feature_from_df_to_df,
    add_static_feature_from_cube_to_df,
    get_training_period_cycles,
    identify_parquet_type,
    split_netcdf_parquet_pickle,
)
from improver.calibration.quantile_regression_random_forest import (
    TrainQuantileRegressionRandomForests,
    quantile_forest_package_available,
)
from improver.memprofile import memray_stage

try:
    from quantile_forest import RandomForestQuantileRegressor
except ModuleNotFoundError:
    # Define empty class to avoid type hint errors.
    class RandomForestQuantileRegressor:
        pass


class LoadForTrainQRF(PostProcessingPlugin):
    """Plugin to load input files for training a Quantile Regression Random Forest
    (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        parquet_diagnostic_names: Union[list[str], str],
        cf_names: Union[list[str], str],
        forecast_periods: str,
        cycletime: str,
        training_length: int,
        experiments: list[str],
        unique_site_id_keys: Union[list[str], str] = "wmo_id",
    ):
        """Initialise the LoadForTrainQRF plugin.

        Args:
            feature_config: Feature configuration defining the features to be used for
                Quantile Regression Random Forests.
            parquet_diagnostic_names: A list containing the diagnostic names that will
                be used for filtering the forecast and truth DataFrames read in from
                the parquet files. The target diagnostic name is expected to be the
                first item in the list. These names could be different from the
                CF name e.g. 'temperature_at_screen_level'. This is expected to be the
                same length as the cf_names and experiments lists.
            cf_names: A list containing the CF names of the diagnostics. The CF names should
                match the order of the parquet_diagnostic_names. The target diagnostic to be
                calibrated is expected to be the first item in the list. These names
                could be different from the diagnostic name used to identify in the
                parquet files. For example, the diagnostic name could be
                'temperature_at_screen_level' and the corresponding CF name could be
                'air_temperature'. This is expected to be the same length as the
                parquet_diagnostic_names and experiments lists.
            forecast_periods: Range of forecast periods to be calibrated in hours in
                the form: "start:end:interval" e.g. "6:18:6" or a single forecast period
                e.g. "6". Multiple ranges can be specified using semicolon separation,
                e.g. "1:133:1;135:199:3" for hourly T+1 to T+132 and
                3-hourly T+135 to T+198.
            cycletime: The time at which the forecast is valid in the form:
                YYYYMMDDTHHMMZ.
            training_length: The number of days of training data to use.
                experiment: The name of the experiment (step) that calibration is applied to.
            experiments: The names of the experiment (step) that calibration is
                applied to. This is used to filter the forecast DataFrame on load.
                This is expected to be the same length as the parquet_diagnostic_names
                and cf_names lists.
            unique_site_id_key: The names of the coordinates that uniquely identify
                each site, e.g. "wmo_id" or "latitude,longitude".
        """
        self.quantile_forest_installed = quantile_forest_package_available()
        self.feature_config = feature_config
        self.parquet_diagnostic_names = parquet_diagnostic_names
        self.cf_names = cf_names
        self.forecast_periods = forecast_periods
        self.cycletime = cycletime
        self.training_length = training_length
        self.experiments = experiments
        if isinstance(unique_site_id_keys, str):
            unique_site_id_keys = [unique_site_id_keys]
        self.unique_site_id_keys = unique_site_id_keys

        if (
            len(set([len(parquet_diagnostic_names), len(cf_names), len(experiments)]))
            > 1
        ):
            msg = (
                "The length of the parquet_diagnostic_names list must match the length "
                "of the cf_names list and the length of the experiments list. "
                "The lengths provided are "
                f"{len(parquet_diagnostic_names)}, {len(cf_names)} and "
                f"{len(experiments)}, respectively."
            )
            raise ValueError(msg)

    def _parse_forecast_periods(self) -> list[int]:
        """Parse the forecast periods argument to produce a list of forecast periods
        in seconds.

        Returns:
            List of forecast periods in seconds, deduplicated and sorted.
        Raises:
            ValueError: If the forecast_periods argument is not a single integer,
                a range in the form 'start:end:interval', or semicolon-separated
                ranges.
        """
        all_periods = []

        # Support semicolon-separated ranges
        tokens = self.forecast_periods.split(";")

        for token in tokens:
            token = token.strip()
            if ":" in token:
                parts = list(map(int, token.split(":")))
                if len(parts) != 3:
                    msg = (
                        "Each forecast_periods range must be in the form "
                        "'start:end:interval' with exactly 3 parts. "
                        f"Invalid range provided: {token}"
                    )
                    raise ValueError(msg)
                all_periods.extend(range(*parts))
            else:
                try:
                    all_periods.append(int(token))
                except ValueError:
                    msg = (
                        "The forecast_periods argument must be a single integer, "
                        "a range in the form 'start:end:interval', "
                        "or semicolon-separated ranges e.g. '1:133:1;135:199:3'. "
                        f"The forecast period provided was: {self.forecast_periods}."
                    )
                    raise ValueError(msg)

        # Deduplicate and sort, then convert to seconds
        forecast_periods = sorted(set(all_periods))
        return [fp * 3600 for fp in forecast_periods]

    def _read_and_normalise_parquet(
        self,
        table_path: pathlib.Path | str,
        filters: list,
        columns: list[str],
        schema,
    ) -> pd.DataFrame:
        """Read parquet data, normalise site IDs and time dtypes, and return a
        pandas DataFrame. Normalisation is performed in pyarrow before conversion
        to pandas. Doing this in pyarrow avoids
        creating a Python string object per row and avoids pandas copying every
        column again afterwards.

        The following are applied:
            - Whitespace is stripped from string-valued unique site ID columns,
              so that additional whitespace doesn't prevent sites matching
              between different forecast inputs and between the forecast and
              truth data (e.g. "03003" vs "03003   ").
            - "time" and "forecast_reference_time" are converted to
              datetime64[ns, UTC].
            - "forecast_period", stored as int64 nanoseconds, is converted to
              timedelta64[ns].

        Args:
            table_path: Path to the parquet dataset.
            filters: Row filters passed to the parquet reader.
            columns: Names of the columns to load.
            schema: pyarrow schema used to read the dataset.

        Returns:
            DataFrame containing the requested rows and columns, with stripped
            string site IDs and nanosecond-resolution time and duration columns.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        # Read only requested columns and matching rows using the supplied schema.
        table = pq.read_table(
            table_path, columns=columns, filters=filters, schema=schema
        )
        for name in table.column_names:
            field = table.schema.field(name)
            if name in self.unique_site_id_keys and (
                pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
            ):
                # Strip site-ID whitespace without changing the string type.
                column = pc.utf8_trim_whitespace(table[name])
            elif name in ["time", "forecast_reference_time"]:
                # Cast timestamps to nanosecond resolution with the UTC timezone.
                column = table[name].cast(pa.timestamp("ns", "UTC"))
            elif name == "forecast_period":
                # Interpret integer nanoseconds as a nanosecond-resolution duration.
                column = table[name].cast(pa.duration("ns"))
            else:
                # Leave columns that do not need normalisation unchanged.
                continue
            # Replace the column at its existing position with the normalised values.
            table = table.set_column(table.schema.get_field_index(name), name, column)
        return table.to_pandas()

    def _diagnostic_merge_columns(
        self, representation: Optional[str] = None
    ) -> list[str]:
        """Return the columns used to merge diagnostics.

        Site, cycle and lead-time columns identify a forecast case. The
        representation column, when supplied, distinguishes percentile or
        realization rows within each case. The case columns are also the
        grouping keys used when ensemble features are aggregated.

        Returns:
            Merge columns for joining one diagnostic to another.
        """
        columns = [
            *self.unique_site_id_keys,
            "forecast_reference_time",
            "forecast_period",
        ]
        if representation is not None:
            columns.append(representation)
        return columns

    @staticmethod
    def _ordered_unique_columns(*column_groups: list[str]) -> list[str]:
        """Combine column groups in order, retaining only their first occurrence."""
        columns = []
        for group in column_groups:
            for column in group:
                if column not in columns:
                    columns.append(column)
        return columns

    def _forecast_feature_columns_to_load(self, schema_names: list[str]) -> list[str]:
        """Select configured features that are stored directly in forecast parquet."""
        return [
            feature
            for features in self.feature_config.values()
            for feature in features
            if feature in schema_names
        ]

    def _forecast_columns_to_load(
        self, representation: str, schema_names: list[str]
    ) -> list[str]:
        """Select columns for the first forecast diagnostic parquet read.

        Args:
            representation: Name of the column distinguishing members within a
                case, either "percentile" or "realization".
            schema_names: Names of the columns available in the forecast schema.

        Returns:
            Site and validity-time keys, diagnostic merge keys, values and
            configured feature columns, without duplicates.
        """
        # Validity time supports truth matching and time-based features.
        validity_columns = [*self.unique_site_id_keys, "time"]
        # Case keys support diagnostic merges and aggregate member features.
        diagnostic_case_columns = self._diagnostic_merge_columns()
        # Each parquet forecast value is renamed to its configured cf_name.
        forecast_value_columns = ["forecast"]
        # Keep the representation after the value column, matching the existing
        # forecast DataFrame column order.
        representation_columns = [representation]
        # Coordinates such as altitude are read; computed features are not columns.
        forecast_feature_columns = self._forecast_feature_columns_to_load(schema_names)

        return self._ordered_unique_columns(
            validity_columns,
            diagnostic_case_columns,
            forecast_value_columns,
            representation_columns,
            forecast_feature_columns,
        )

    def _predictor_columns_to_load(self, representation: str) -> list[str]:
        """Select only merge keys and values for subsequent diagnostic reads."""
        return self._ordered_unique_columns(
            self._diagnostic_merge_columns(representation),
            ["forecast"],
        )

    def _truth_columns_to_load(self) -> list[str]:
        """Select site/time join keys and the observation value from parquet."""
        return self._ordered_unique_columns(
            self.unique_site_id_keys,
            ["time"],
            ["ob_value"],
        )

    def _read_parquet_files(
        self,
        forecast_table_path: pathlib.Path | str,
        truth_table_path: pathlib.Path | str,
        forecast_periods: list[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read the forecast and truth data from parquet files.
        self.quantile_forest_installed = quantile_forest_package_available()
        Args:
            forecast_table_path: Path to the forecast parquet file.
            truth_table_path: Path to the truth parquet file.
            forecast_periods: List of forecast periods in seconds.

        Returns:
            Tuple containing:
                - DataFrame containing the forecast data.
                - DataFrame containing the truth data.

        Raises:
            ValueError: If the forecast parquet file does not contain the expected
                fields. Either "percentile" or "realization".
            ValueError: If the forecast parquet file does not contain the expected
                features.
            ValueError: If the truth parquet file does not contain the expected
                fields.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        cycletimes = []
        for forecast_period in forecast_periods:
            # Load forecasts from parquet file filtering by diagnostic and blend_time.
            cycletimes.extend(
                get_training_period_cycles(
                    self.cycletime, forecast_period, self.training_length
                )
            )
        cycletimes = list(set(cycletimes))

        parquet_files = sorted(Path(forecast_table_path).glob("**/*.parquet"))
        altered_schema = None
        for parquet_file in parquet_files:
            schema = pq.read_schema(parquet_file)
            if schema.get_all_field_indices("percentile"):
                altered_schema = CalibrationSchemas().FORECAST_SCHEMA
                break
            if schema.get_all_field_indices("realization"):
                altered_schema = CalibrationSchemas().FORECAST_SCHEMA.remove(
                    CalibrationSchemas().FORECAST_SCHEMA.get_field_index("percentile")
                )
                altered_schema = altered_schema.append(
                    pa.field("realization", pa.int64())
                )
                break

        if altered_schema is None:
            msg = (
                "The forecast parquet files are expected to contain either a "
                "'percentile' or 'realization' field. Neither was found in any "
                f"candidate file under: {forecast_table_path}."
            )
            raise ValueError(msg)

        representation_name = (
            "percentile" if "percentile" in altered_schema.names else "realization"
        )
        # The first diagnostic supplies shared features and validity time;
        # later diagnostics supply values matched on diagnostic merge columns.
        diagnostic_merge_columns = self._diagnostic_merge_columns(representation_name)
        initial_forecast_columns = self._forecast_columns_to_load(
            representation_name, altered_schema.names
        )
        predictor_columns = self._predictor_columns_to_load(representation_name)
        # Forecast periods are stored as int64 nanoseconds.
        forecast_periods_ns = sorted({int(fp * 1e9) for fp in forecast_periods})

        forecast_df = None
        for parquet_diagnostic_name, cf_name, experiment in zip(
            self.parquet_diagnostic_names, self.cf_names, self.experiments
        ):
            filters = [
                [
                    ("diagnostic", "==", parquet_diagnostic_name),
                    ("blend_time", "in", cycletimes),
                    ("forecast_period", "in", forecast_periods_ns),
                    ("experiment", "==", experiment),
                ]
            ]

            additional_df = self._read_and_normalise_parquet(
                forecast_table_path,
                filters,
                (
                    initial_forecast_columns
                    if forecast_df is None
                    else predictor_columns
                ),
                altered_schema,
            )

            if additional_df.empty:
                if forecast_df is None:
                    return None, None
                msg = (
                    "The requested parquet diagnostic name is not present in the "
                    f"forecast parquet file: {parquet_diagnostic_name}."
                )
                raise ValueError(msg)

            if forecast_df is None:
                additional_df.rename(columns={"forecast": cf_name}, inplace=True)
                forecast_df = additional_df
                continue

            representation = (
                "percentile" if "percentile" in additional_df.columns else "realization"
            )
            merge_on = diagnostic_merge_columns.copy()

            # If e.g. percentile is all NaN as this is a deterministic diagnostic, remove this column.
            if additional_df[representation].isna().all():
                merge_on.remove(representation)

            forecast_df = add_feature_from_df_to_df(
                forecast_df,
                additional_df.rename(columns={"forecast": cf_name}),
                cf_name,
                merge_on,
            )

        forecast_df = forecast_df.reset_index(drop=True)

        # Load truths from parquet file filtering by diagnostic.
        filters = [[("diagnostic", "==", self.parquet_diagnostic_names[0])]]
        truth_df = self._read_and_normalise_parquet(
            truth_table_path,
            filters,
            self._truth_columns_to_load(),
            CalibrationSchemas().TRUTH_SCHEMA,
        )

        if truth_df.empty:
            msg = (
                f"The requested filepath {truth_table_path} does not contain the "
                f"requested contents: {filters}"
            )
            raise IOError(msg)
        return forecast_df, truth_df

    def process(
        self,
        file_paths: list[pathlib.Path | str],
    ) -> Optional[tuple[iris.cube.CubeList, pathlib.Path | str, pathlib.Path | str]]:
        """Load input files for training a Quantile Regression Random Forest (QRF)
        model. Two sources of input data must be provided: historical forecasts and
        historical truth data (to use in calibration).

        Args:
            file_paths (cli.inputpaths):
                A list of input paths containing:
                - The path to a Parquet file containing the truths to be used
                for calibration. The expected columns within the
                Parquet file are: ob_value, time, wmo_id, diagnostic, latitude,
                longitude and altitude.
                - The path to a Parquet file containing the forecasts to be used
                for calibration. The expected columns within the Parquet file are:
                forecast, blend_time, forecast_period, forecast_reference_time, time,
                wmo_id, percentile, diagnostic, latitude, longitude, period, height,
                cf_name, units. Please note that the presence of a forecast_period
                column is used to separate the forecast parquet file from the truth
                parquet file.
                - Optionally, paths to NetCDF files containing additional predictors.

        Returns:
            Tuple containing:
                - DataFrame containing the forecast data.
                - DataFrame containing the truth data.
                - List of cubes containing additional features.

            A tuple of (None, None, None) is returned if:
                - The quantile_forest package is not installed.
                - No parquet files are provided.
                - Either the forecast or truth parquet files are missing.

        """
        if not self.quantile_forest_installed:
            return None, None, None
        cube_inputs, parquets, _ = split_netcdf_parquet_pickle(file_paths)

        # If there are no parquet files, return None.
        if not parquets:
            msg = "No parquet files have been provided."
            warnings.warn(msg)
            return None, None, None

        forecast_table_path, truth_table_path = identify_parquet_type(parquets)

        # If either the forecast or truth parquet files are missing, return None.
        if not forecast_table_path or not truth_table_path:
            msg = "Both forecast and truth parquet files must be provided."
            warnings.warn(msg)
            return None, None, None

        forecast_periods = self._parse_forecast_periods()
        forecast_df, truth_df = self._read_parquet_files(
            forecast_table_path, truth_table_path, forecast_periods
        )
        if forecast_df is None:
            msg = (
                "The forecast parquet files are empty after filtering. "
                "Unable to train the Quantile Regression Random Forest model."
            )
            warnings.warn(msg)
            return None, None, None

        if cube_inputs is None:
            cube_inputs = iris.cube.CubeList()
        missing_features = (
            set(self.feature_config.keys())
            - set(forecast_df.columns)
            - set([c.name() for c in cube_inputs])
        )

        if len(missing_features) > 0:
            msg = (
                "The features requested in the feature_config are absent from "
                "the forecast parquet file and the input cubes. "
                f"The missing fields are: {missing_features}."
            )
            raise ValueError(msg)
        return forecast_df, truth_df, cube_inputs


class PrepareAndTrainQRF(PostProcessingPlugin):
    """Plugin to prepare and train a Quantile Regression Random Forest (QRF) model."""

    def __init__(
        self,
        feature_config: dict[str, list[str]],
        target_cf_name: str,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_samples: Optional[float] = None,
        random_state: Optional[int] = None,
        transformation: Optional[str] = None,
        pre_transform_addition: float = 0,
        unique_site_id_keys: Union[list[str], str] = "wmo_id",
        **kwargs,
    ):
        """Initialise the PrepareAndTrainQRF plugin.

        Args:
            feature_config: Feature configuration defining the features to be used for
                Quantile Regression Random Forests.
            target_cf_name: A string containing the CF name of the forecast to be
                calibrated e.g. air_temperature.
            n_estimators: The number of trees in the forest.
            max_depth: The maximum depth of the trees.
            max_samples: The maximum number of samples to draw from the total number of
                samples to train each tree.
            random_state: Seed used by the random number generator.
            transformation: Transformation to be applied to the data before fitting.
            pre_transform_addition: Value to be added before transformation.
            unique_site_id_key: The names of the coordinates that uniquely identify
                each site, e.g. "wmo_id" or ["latitude", "longitude"].
            kwargs: Additional keyword arguments for the quantile regression model.
        """
        self.feature_config = feature_config
        self.target_cf_name = target_cf_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.random_state = random_state
        self.transformation = transformation
        self.pre_transform_addition = pre_transform_addition
        if isinstance(unique_site_id_keys, str):
            unique_site_id_keys = [unique_site_id_keys]
        self.unique_site_id_keys = unique_site_id_keys
        self.kwargs = kwargs
        self.quantile_forest_installed = quantile_forest_package_available()
        self.float_decimals = 4

    @staticmethod
    def _check_matching_times(
        forecast_df: pd.DataFrame, truth_df: pd.DataFrame
    ) -> list[pd.Timestamp]:
        """Find the intersecting times available within the forecast and truth
        DataFrames.

        Args:
            forecast_df: DataFrame containing the forecast data.
            truth_df: DataFrame containing the truth data.
        Returns:
            List of intersecting times as pandas Timestamp objects.
        """
        # Calling unique() on the time column is quicker than relying upon set() to
        # find the unique times.
        return list(
            set(forecast_df["time"].unique()).intersection(
                set(truth_df["time"].unique())
            )
        )

    def _add_static_features_from_cubes_to_df(
        self, forecast_df: pd.DataFrame, cube_inputs: iris.cube.CubeList
    ) -> pd.DataFrame:
        """Add features to the forecast DataFrame from cubes based on the feature
        configuration. Other features are expected to already be present in the
        forecast DataFrame.

        Args:
            forecast_df: DataFrame containing the forecast data.
            cube_inputs: List of cubes containing additional features.

        Returns:
            DataFrame with additional features added from the input cubes.
        """
        if cube_inputs is None or len(cube_inputs) == 0:
            return forecast_df

        for feature_name, feature_list in self.feature_config.items():
            for feature in feature_list:
                if feature == "static":
                    # Use the cube's data directly as a feature.
                    constr = iris.Constraint(name=feature_name)
                    # Static features can be provided either as a cube or as a column
                    # in the forecast DataFrame.
                    try:
                        feature_cube = cube_inputs.extract_cube(constr)
                    except iris.exceptions.ConstraintMismatchError:
                        feature_cube = None
                    if not feature_cube:
                        continue
                    forecast_df = add_static_feature_from_cube_to_df(
                        forecast_df,
                        feature_cube,
                        feature_name,
                        [*self.unique_site_id_keys],
                        float_decimals=self.float_decimals,
                    )
        return forecast_df

    def filter_bad_sites(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Remove sites that have NaNs in the data.

        Args:
            feature_df: DataFrame containing the forecast data with features.
            truth_df: DataFrame containing the truth data.

        Returns:
            Tuple containing:
                - DataFrame containing the forecast data with bad sites removed.
                - DataFrame containing the truth data with bad sites removed.

        Raises:
            ValueError: If the truth DataFrame is empty after removing NaNs.
            ValueError: If there are no matching sites and times between the
                forecast and truth DataFrames after removing NaNs.
        """
        truth_df.dropna(subset=["ob_value"] + [*self.unique_site_id_keys], inplace=True)

        if truth_df.empty:
            msg = "Empty truth DataFrame after removing NaNs."
            raise ValueError(msg)

        # Match sites and validity times without copying unrelated feature columns.
        match_columns = [*self.unique_site_id_keys, "time"]
        forecast_index = pd.MultiIndex.from_frame(forecast_df[match_columns])
        truth_index = pd.MultiIndex.from_frame(truth_df[match_columns])
        forecast_df = forecast_df[forecast_index.isin(truth_index)]
        truth_df = truth_df[truth_index.isin(forecast_index)]

        if truth_df.empty:
            msg = (
                "Empty truth DataFrame after finding the intersection of sites "
                "and times between the truth DataFrame and the forecast DataFrame."
            )
            raise ValueError(msg)

        return forecast_df, truth_df

    def process(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
        cube_inputs: Optional[iris.cube.CubeList] = None,
    ) -> Optional[tuple[RandomForestQuantileRegressor, str, float]]:
        """Load input files and train a Quantile Regression Random Forest (QRF)
        model. This model can be applied later to calibrate the forecast. Two sources
        of input data must be provided: historical forecasts and historical truth data
        (to use in calibration). The model is output as a pickle file.

        This compatibility interface also fits the model. To release caller-owned
        DataFrames and cubes before fitting, call :meth:`prepare_training_data`,
        release the input references, then call :meth:`fit_training_data`.

        Args:
            forecast_df: DataFrame containing the forecast data.
            truth_df: DataFrame containing the truth data.
            cube_inputs: List of cubes containing additional features.

        Returns: A tuple containing:
            - The trained RandomForestQuantileRegressor model.
            - The transformation applied to the data before fitting.
            - The value added before transformation.

        Raises:
            ValueError: If there are no matching times between the forecast and truth
                data.
        """
        training_data = self.prepare_training_data(forecast_df, truth_df, cube_inputs)
        del forecast_df, truth_df, cube_inputs
        if training_data is None:
            return None, None, None
        return self.fit_training_data(*training_data)

    def prepare_training_data(
        self,
        forecast_df: pd.DataFrame,
        truth_df: pd.DataFrame,
        cube_inputs: Optional[iris.cube.CubeList] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Add static features, filter sites and prepare arrays without fitting.

        Args:
            forecast_df: Forecast information and features.
            truth_df: Observations used for calibration.
            cube_inputs: Cubes containing additional static features.

        Returns:
            Independent feature and target arrays, or None if training cannot
            proceed. Callers should release their DataFrame and cube references
            before passing the arrays to :meth:`fit_training_data`. This method
            does not retain the inputs on the plugin.
        """
        if not self.quantile_forest_installed:
            return None

        intersecting_times = self._check_matching_times(forecast_df, truth_df)
        if len(intersecting_times) == 0:
            msg = (
                "No matching times between the forecast and truth data. "
                "Unable to train the Quantile Regression Random Forest model."
            )
            warnings.warn(msg)
            return None

        with memray_stage("static_features"):
            forecast_df = self._add_static_features_from_cubes_to_df(
                forecast_df, cube_inputs
            )
            forecast_df, truth_df = self.filter_bad_sites(forecast_df, truth_df)

        return self._training_plugin().build_training_arrays(forecast_df, truth_df)

    def _training_plugin(self) -> TrainQuantileRegressionRandomForests:
        """Create a trainer with the preparation and fitting configuration."""
        return TrainQuantileRegressionRandomForests(
            target_name=self.target_cf_name,
            feature_config=self.feature_config,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_samples=self.max_samples,
            random_state=self.random_state,
            transformation=self.transformation,
            pre_transform_addition=self.pre_transform_addition,
            unique_site_id_keys=self.unique_site_id_keys,
            **self.kwargs,
        )

    def fit_training_data(
        self, feature_values: np.ndarray, target_values: np.ndarray
    ) -> tuple[RandomForestQuantileRegressor, Optional[str], float]:
        """Fit prepared arrays and return the existing serialisable model tuple.

        Args:
            feature_values: Feature array with shape (samples, features).
            target_values: Target array with shape (samples,).

        Returns:
            The trained model, transformation and pre-transform addition, as
            returned by :meth:`process` and consumed by the application code.
        """
        with memray_stage("fitting"):
            result = self._training_plugin().fit_qrf(feature_values, target_values)
        return (result, self.transformation, self.pre_transform_addition)

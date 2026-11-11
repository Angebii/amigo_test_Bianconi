# # Bias correction of the CMIP6 SSP5-8.5 projections
## Requires:
##   DATA/CMIP6/<model>/historical/<variable>/*.nc   (daily tasmax, tasmin, pr, calibration 2000-2014)
##   DATA/CMIP6/<model>/ssp585/<variable>/*.nc       (daily tasmax, tasmin, pr, 2015-2090)
##   DATA/ERA5-Land/<variable folders>/*.nc           (daily gridded ERA5-Land: grid, land cells, calibration)
##   DATA/OTHERS/france_regions_2016.gpkg             (region boundaries)
##   data_processed/era5_daily_regional.csv           (notebook/03_climate_indices.py)
##   data_processed/cmip6_historical_regional.csv     (notebook/07_cmip6_distributions.py)
##   data_processed/baseline_dataset.csv              (notebook/03_climate_indices.py)
##   results/baseline/config.json                     (notebook/04_baseline.py: indices, interaction, penalty)
##
## Two future windows of 20 harvest seasons, as long as the baseline of the yields (2001-2019):
##   2031-2050: from 1 October 2030 (sowing of the 2031 season) to 31 July 2050
##   2071-2090: from 1 October 2070 (sowing of the 2071 season) to 31 July 2090
##
## For every model and variable, the correction is trained once (train_bias_correction): quantiles of
## ERA5-Land and of the raw historical simulation of the same model over 2000-2014, for every grid cell
## and calendar month. It is then applied (apply_bias_correction) to the SSP5-8.5 data of both windows,
## interpolated on the ERA5-Land grid.
## Each window is corrected on its own, so that quantile delta mapping keeps the change simulated
## by the model between 2000-2014 and that window. Then regional means, as for ERA5-Land.

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from amigo_yield import prepare_fold, prepare_inputs, compute_season_indices, INDEX_NAMES, read_era5, build_region_mask, compute_regional_means, read_cmip6_variable, interpolate_to_grid, train_bias_correction, apply_bias_correction, CMIP6_NAMES

CMIP6_DIR = Path("DATA/CMIP6")
ERA5_DIR = Path("DATA/ERA5-Land")
BOUNDARIES_FILE = Path("DATA/OTHERS/france_regions_2016.gpkg")
PROCESSED_DIR = Path("data_processed")
FUTURE_RAW_FILE = PROCESSED_DIR / "cmip6_ssp585_regional.csv"
FUTURE_CORRECTED_FILE = PROCESSED_DIR / "cmip6_ssp585_corrected_regional.csv"

MODELS = ["CNRM-ESM2-1", "EC-Earth3", "IPSL-CM6A-LR"]
VARIABLE_NAMES = ["tmax", "tmin", "precip"]

## Calibration period: common period of the historical simulations and ERA5-Land
CALIBRATION_START = "2000-01-01"
CALIBRATION_END = "2014-12-31"

## Future windows: name -> first and last day (20 harvest seasons each)
WINDOWS = {
	"2031-2050": ("2030-10-01", "2050-07-31"),
	"2071-2090": ("2070-10-01", "2090-07-31"),
}

## Present: the 19 harvest seasons of the yields used to train the Ridge regression,
## from 1 October 2000 (sowing of the 2001 season) to 31 July 2019
PRESENT_START = "2000-10-01"
PRESENT_END = "2019-07-31"

## Months of the growing season
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]

## Percentiles of the daily distributions
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

## Chunks of the grids for the bias correction: whole time series, blocks of cells
## (xsdba needs the whole series of every cell; smaller blocks use less memory)
CHUNKS = {"time": -1, "latitude": 24, "longitude": 36}


# %% ERA5-Land daily grids of the calibration period, grid, land cells and region mask
era5_variables, _ = read_era5(ERA5_DIR, VARIABLE_NAMES)
for variable in VARIABLE_NAMES:
	era5_variables[variable] = era5_variables[variable].sel(time=slice(CALIBRATION_START, CALIBRATION_END))

first_day = era5_variables["tmax"].isel(time=0)
first_day = first_day.drop_vars(["time", "spatial_ref"], errors="ignore")
latitudes = first_day["latitude"]
longitudes = first_day["longitude"]
land_mask = first_day.notnull()
region_mask = build_region_mask(BOUNDARIES_FILE, longitudes, latitudes)
print("grid:", len(latitudes), "x", len(longitudes), "| land cells:", int(land_mask.sum()))


# %% Correction of every model and window, then regional means of the raw and of the corrected grids.
## The results are saved to CSV files: if both files exist, they are read instead of recomputed.
## Recomputing: the correction is trained 9 times (3 models x 3 variables) and applied 18 times (x 2 windows).
if FUTURE_RAW_FILE.exists() and FUTURE_CORRECTED_FILE.exists():
	print("reading", FUTURE_RAW_FILE, "and", FUTURE_CORRECTED_FILE)
	future_daily = pd.read_csv(FUTURE_RAW_FILE, parse_dates=["date"])
	future_corrected_daily = pd.read_csv(FUTURE_CORRECTED_FILE, parse_dates=["date"])
else:
	raw_tables = []
	corrected_tables = []
	for model in MODELS:

		## Raw and corrected grids of every window: window -> variable -> grid
		raw_variables = {}
		corrected_variables = {}
		for window in WINDOWS:
			raw_variables[window] = {}
			corrected_variables[window] = {}

		for cmip6_name in CMIP6_NAMES:
			## Training, once per model and variable: ERA5-Land and the raw historical simulation, 2000-2014
			historical = read_cmip6_variable(CMIP6_DIR, model, "historical", cmip6_name, CALIBRATION_START, CALIBRATION_END)
			variable = historical.name
			historical = interpolate_to_grid(historical, latitudes, longitudes, land_mask).chunk(CHUNKS)
			observed = era5_variables[variable].drop_vars("spatial_ref", errors="ignore").chunk(CHUNKS)

			print(model, variable, "training")
			adjustment = train_bias_correction(observed, historical, variable)

			## Application to every window of SSP5-8.5
			for window, (start_date, end_date) in WINDOWS.items():
				future = read_cmip6_variable(CMIP6_DIR, model, "ssp585", cmip6_name, start_date, end_date)
				future = interpolate_to_grid(future, latitudes, longitudes, land_mask).chunk(CHUNKS)

				raw_variables[window][variable] = future
				corrected_variables[window][variable] = apply_bias_correction(adjustment, future, variable)

		## Regional means of every window
		for window in WINDOWS:
			print(model, window, "regional means")
			raw_table = compute_regional_means(raw_variables[window], region_mask)
			raw_table["source"] = model
			raw_table["window"] = window
			raw_tables.append(raw_table)

			corrected_table = compute_regional_means(corrected_variables[window], region_mask)
			corrected_table["source"] = model + " corrected"
			corrected_table["window"] = window
			corrected_tables.append(corrected_table)

	future_daily = pd.concat(raw_tables, ignore_index=True)
	future_corrected_daily = pd.concat(corrected_tables, ignore_index=True)
	future_daily.to_csv(FUTURE_RAW_FILE, index=False)
	future_corrected_daily.to_csv(FUTURE_CORRECTED_FILE, index=False)
	print("saved", FUTURE_RAW_FILE, "and", FUTURE_CORRECTED_FILE)


# %% Check: dates and number of days of every model and window
check = future_corrected_daily.groupby(["source", "window"])["date"].agg(["min", "max", "count"])
print(check)


# %% Check: quantile delta mapping keeps the change simulated by every model.
## Change of the mean of the growing season (October to July) between 2000-2014 and every window:
##   raw:       raw future minus raw historical of the same model
##   corrected: corrected future minus ERA5-Land (the corrected historical has the ERA5-Land distribution)
## Temperature in degC, precipitation in %. The two columns of every variable should be close.
ERA5_DAILY_FILE = PROCESSED_DIR / "era5_daily_regional.csv"
CMIP6_HISTORICAL_FILE = PROCESSED_DIR / "cmip6_historical_regional.csv"

era5_daily = pd.read_csv(ERA5_DAILY_FILE, parse_dates=["date"])
historical_daily = pd.read_csv(CMIP6_HISTORICAL_FILE, parse_dates=["date"])

## Growing-season means of the calibration period
is_calibration = (era5_daily["date"] >= CALIBRATION_START) & (era5_daily["date"] <= CALIBRATION_END)
is_season = era5_daily["date"].dt.month.isin(SEASON_MONTHS)
era5_means = era5_daily.loc[is_calibration & is_season, VARIABLE_NAMES].mean()

is_season = historical_daily["date"].dt.month.isin(SEASON_MONTHS)
historical_means = historical_daily[is_season].groupby("source")[VARIABLE_NAMES].mean()

change_rows = []
for model in MODELS:
	for window in WINDOWS:
		is_raw = (future_daily["source"] == model) & (future_daily["window"] == window) & future_daily["date"].dt.month.isin(SEASON_MONTHS)
		is_corrected = (future_corrected_daily["source"] == model + " corrected") & (future_corrected_daily["window"] == window) & future_corrected_daily["date"].dt.month.isin(SEASON_MONTHS)
		raw_means = future_daily.loc[is_raw, VARIABLE_NAMES].mean()
		corrected_means = future_corrected_daily.loc[is_corrected, VARIABLE_NAMES].mean()

		row = {"model": model, "window": window}
		for variable in ["tmax", "tmin"]:
			row[variable + " raw"] = raw_means[variable] - historical_means.loc[model, variable]
			row[variable + " corrected"] = corrected_means[variable] - era5_means[variable]
		row["precip % raw"] = (raw_means["precip"] / historical_means.loc[model, "precip"] - 1) * 100
		row["precip % corrected"] = (corrected_means["precip"] / era5_means["precip"] - 1) * 100
		change_rows.append(row)

changes = pd.DataFrame(change_rows)
print(changes.round(1).to_string(index=False))


# %% Daily distributions: ERA5-Land of the present (the data of the Ridge regression) and the
## corrected models of every future window, before any index is computed.
## Present: ERA5-Land, harvest seasons 2001-2019. Future: every corrected model and the multi-model
## mean of the corrected daily values ((CNRM-ESM2-1 + EC-Earth3 + IPSL-CM6A-LR) / 3 for every region
## and day), windows 2031-2050 and 2071-2090. Daily regional values of the growing season only.
is_present = (era5_daily["date"] >= PRESENT_START) & (era5_daily["date"] <= PRESENT_END)
present_daily = era5_daily[is_present].copy()
present_daily["source"] = "ERA5-Land"
present_daily["window"] = "2001-2019"

## Multi-model mean of the corrected daily values, window by window
multi_model_future = future_corrected_daily.groupby(["window", "cod_regione", "date"])[VARIABLE_NAMES].mean().reset_index()
multi_model_future["source"] = "multi-model mean corrected"

## All sources in one table, growing season only
comparison_daily = pd.concat([present_daily, future_corrected_daily, multi_model_future], ignore_index=True)
is_in_season = comparison_daily["date"].dt.month.isin(SEASON_MONTHS)
comparison_daily = comparison_daily[is_in_season]
print(comparison_daily.groupby(["window", "source"])["date"].agg(["min", "max", "count"]))


# %% Statistics of the daily distribution of every source and window:
## mean, standard deviation, minimum, percentiles and maximum of all the daily regional values
statistic_names = ["mean", "std", "min"] + ["p" + str(p) for p in PERCENTILES] + ["max"]
CORRECTED_MODELS = [model + " corrected" for model in MODELS]

## Order of the rows: present, then every window with the three models and their mean
source_windows = [("ERA5-Land", "2001-2019")]
for window in WINDOWS:
	for source in CORRECTED_MODELS + ["multi-model mean corrected"]:
		source_windows.append((source, window))

statistic_rows = []
for source, window in source_windows:
	is_source = (comparison_daily["source"] == source) & (comparison_daily["window"] == window)
	source_days = comparison_daily[is_source]

	for variable in VARIABLE_NAMES:
		values = source_days[variable].dropna().to_numpy()

		row = {"source": source, "window": window, "variable": variable}
		row["mean"] = values.mean()
		row["std"] = values.std()
		row["min"] = values.min()
		for p in PERCENTILES:
			row["p" + str(p)] = np.percentile(values, p)
		row["max"] = values.max()
		statistic_rows.append(row)

future_statistics = pd.DataFrame(statistic_rows)

for variable in VARIABLE_NAMES:
	table = future_statistics[future_statistics["variable"] == variable]
	table = table.set_index(["window", "source"])[statistic_names]
	print()
	print(variable)
	print(table.round(1).to_string())


# %% Change from the present: every future source minus ERA5-Land 2001-2019, statistic by statistic.
## Temperature in degC, precipitation in mm per day.
present_statistics = future_statistics[future_statistics["source"] == "ERA5-Land"].set_index("variable")[statistic_names]

change_rows = []
for _, row in future_statistics.iterrows():
	if row["source"] == "ERA5-Land":
		continue

	change = {"source": row["source"], "window": row["window"], "variable": row["variable"]}
	for name in statistic_names:
		change[name] = row[name] - present_statistics.loc[row["variable"], name]
	change_rows.append(change)

future_changes = pd.DataFrame(change_rows)

for variable in VARIABLE_NAMES:
	table = future_changes[future_changes["variable"] == variable]
	table = table.set_index(["window", "source"])[statistic_names]
	print()
	print(variable, "minus ERA5-Land 2001-2019")
	print(table.round(1).to_string())


# %% Save the tables
RESULTS_DIR = Path("results") / "cmip6"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
future_statistics.to_csv(RESULTS_DIR / "future_distribution_statistics.csv", index=False)
future_changes.to_csv(RESULTS_DIR / "future_distribution_changes.csv", index=False)
print("saved results in", RESULTS_DIR)


# # Level 1: climate indices of the present and of the future, ensemble on the indices
## The seven indices of the Ridge regression (compute_season_indices, the same function used for
## ERA5-Land in notebook/03_climate_indices.py) are computed on ERA5-Land 2001-2019 and on every
## corrected model in every window. The ensemble is made on the indices: every statistic is computed
## for every model, then averaged over the three models; the lowest and the highest model give the spread.

# %% Indices of every season and region: present (ERA5-Land) and every corrected model and window
present_indices = compute_season_indices(present_daily)
present_indices["source"] = "ERA5-Land"
present_indices["window"] = "2001-2019"

index_tables = [present_indices]
for model in CORRECTED_MODELS:
	for window in WINDOWS:
		is_source = (future_corrected_daily["source"] == model) & (future_corrected_daily["window"] == window)
		model_indices = compute_season_indices(future_corrected_daily[is_source])
		model_indices["source"] = model
		model_indices["window"] = window
		index_tables.append(model_indices)

season_indices = pd.concat(index_tables, ignore_index=True)

## Only complete seasons (the first October of every window is in the data, so all seasons should be complete)
season_indices = season_indices.dropna(subset=INDEX_NAMES)
print(season_indices.groupby(["window", "source"])["harvest_year"].agg(["min", "max", "count"]))

season_indices.to_csv(PROCESSED_DIR / "season_indices_present_future.csv", index=False)


# %% Mean of every index: present, every model, and ensemble (mean of the three models, lowest, highest)
## Mean over all regions and seasons of the window
index_means = season_indices.groupby(["window", "source"])[INDEX_NAMES].mean()

ensemble_rows = []
for window in WINDOWS:
	model_means = index_means.loc[window].loc[CORRECTED_MODELS]

	for summary in ["mean", "min", "max"]:
		row = {"window": window, "source": "ensemble " + summary}
		for name in INDEX_NAMES:
			if summary == "mean":
				row[name] = model_means[name].mean()
			if summary == "min":
				row[name] = model_means[name].min()
			if summary == "max":
				row[name] = model_means[name].max()
		ensemble_rows.append(row)

ensemble_means = pd.DataFrame(ensemble_rows).set_index(["window", "source"])
index_means = pd.concat([index_means, ensemble_means]).sort_index()

## One table per index is easier to read: rows = window and source, columns = indices
print(index_means.round(1).T.to_string())


# %% Frequency of extreme seasons: share of region-seasons beyond the extremes of the present.
## For every region, the thresholds are the 10th and the 90th percentile of the index over the
## present seasons (2001-2019), so by construction 10% of the present seasons are below and 10% above.
## A share above 10% in the future means that seasons extreme today become more frequent.
present_rows = season_indices[season_indices["source"] == "ERA5-Land"]
low_thresholds = present_rows.groupby("cod_regione")[INDEX_NAMES].quantile(0.1)
high_thresholds = present_rows.groupby("cod_regione")[INDEX_NAMES].quantile(0.9)

frequency_rows = []
for (window, source), rows in season_indices.groupby(["window", "source"]):
	row = {"window": window, "source": source}
	for name in INDEX_NAMES:
		low = rows["cod_regione"].map(low_thresholds[name])
		high = rows["cod_regione"].map(high_thresholds[name])
		row[name + " below p10 %"] = (rows[name] < low).mean() * 100
		row[name + " above p90 %"] = (rows[name] > high).mean() * 100
	frequency_rows.append(row)

frequencies = pd.DataFrame(frequency_rows).set_index(["window", "source"])
frequency_names = list(frequencies.columns)

## Ensemble: mean of the three models, lowest and highest model
ensemble_rows = []
for window in WINDOWS:
	model_frequencies = frequencies.loc[window].loc[CORRECTED_MODELS]

	for summary in ["mean", "min", "max"]:
		row = {"window": window, "source": "ensemble " + summary}
		for name in frequency_names:
			if summary == "mean":
				row[name] = model_frequencies[name].mean()
			if summary == "min":
				row[name] = model_frequencies[name].min()
			if summary == "max":
				row[name] = model_frequencies[name].max()
		ensemble_rows.append(row)

ensemble_frequencies = pd.DataFrame(ensemble_rows).set_index(["window", "source"])
frequencies = pd.concat([frequencies, ensemble_frequencies]).sort_index()
print(frequencies.round(0).T.to_string())


# %% Save the tables of level 1
index_means.to_csv(RESULTS_DIR / "index_means_present_future.csv")
frequencies.to_csv(RESULTS_DIR / "index_extreme_frequencies.csv")
print("saved results in", RESULTS_DIR)


# # Level 2: yield anomalies predicted by the Ridge regression of the baseline
## The Ridge regression is fitted once on all the 247 region-seasons 2001-2019, with the indices,
## the interaction and the penalty of the baseline (results/baseline/config.json). It is then applied
## to the indices of every corrected model and window. The indices of the future are standardised
## with the regional means and the standard deviations of 2001-2019 (prepare_inputs), so the model
## sees how far every future season is from the present normal of its region.
## The predictions are anomalies from the technology trend: they give the effect of the climate
## with the technology of today. The ensemble is made on the results of every model.

# %% Configuration of the baseline and final Ridge regression on all the region-seasons
with open(Path("results") / "baseline" / "config.json") as config_file:
	baseline_config = json.load(config_file)

ridge_index_names = baseline_config["index_names"]
ridge_interaction = tuple(baseline_config["interaction"])
ridge_alpha = baseline_config["alpha_all_years"]
LOSS_THRESHOLD = baseline_config["loss_threshold"]
print("indices:", ridge_index_names)
print("interaction:", ridge_interaction, "| alpha:", ridge_alpha, "| loss threshold:", LOSS_THRESHOLD, "%")

baseline_dataset = pd.read_csv(PROCESSED_DIR / "baseline_dataset.csv")
x_all, y_all, _, _ = prepare_fold(baseline_dataset, baseline_dataset, ridge_index_names, ridge_interaction)
final_ridge = Ridge(alpha=ridge_alpha)
final_ridge.fit(x_all, y_all)
print("region-seasons used for the fit:", len(baseline_dataset))


# %% Predicted anomalies of the present and of every corrected model and window
## Present: the same final model applied to the 2001-2019 indices (fitted values), so that present
## and future are predicted in the same way; the observed anomalies are kept for comparison.
present_predictions = baseline_dataset[["cod_regione", "harvest_year"]].copy()
present_predictions["source"] = "ERA5-Land"
present_predictions["window"] = "2001-2019"
present_predictions["observed_anomaly"] = y_all
present_predictions["predicted_anomaly"] = final_ridge.predict(x_all)

## Future: only the regions of the baseline
is_future = season_indices["source"].isin(CORRECTED_MODELS)
is_baseline_region = season_indices["cod_regione"].isin(baseline_dataset["cod_regione"].unique())
future_rows = season_indices[is_future & is_baseline_region].reset_index(drop=True)

_, x_future = prepare_inputs(baseline_dataset, future_rows, ridge_index_names, ridge_interaction)
future_predictions = future_rows[["cod_regione", "harvest_year", "source", "window"]].copy()
future_predictions["predicted_anomaly"] = final_ridge.predict(x_future)

yield_predictions = pd.concat([present_predictions, future_predictions], ignore_index=True)
yield_predictions.to_csv(RESULTS_DIR / "yield_anomaly_predictions.csv", index=False)
print(yield_predictions.groupby(["window", "source"])["harvest_year"].agg(["min", "max", "count"]))


# %% Extrapolation: share of the future region-seasons with an index outside the range of 2001-2019.
## The Ridge regression is linear: outside the values seen in the fit it extends the same slope,
## so these predictions are less reliable.
x_train_inputs, x_future_inputs = prepare_inputs(baseline_dataset, future_rows, ridge_index_names, None)
lowest_present = x_train_inputs.min(axis=0)
highest_present = x_train_inputs.max(axis=0)
is_outside = (x_future_inputs < lowest_present) | (x_future_inputs > highest_present)

outside = pd.DataFrame(is_outside, columns=ridge_index_names)
outside["window"] = future_rows["window"]
outside["source"] = future_rows["source"]
outside["any index"] = is_outside.any(axis=1)
outside_shares = outside.groupby(["window", "source"]).mean() * 100
print(outside_shares.round(0).T.to_string())


# %% Summary of every source and window: mean predicted anomaly and share of loss seasons
## (predicted anomaly below the loss threshold of the baseline), then the ensemble of the three models
summary_rows = []
for (window, source), rows in yield_predictions.groupby(["window", "source"]):
	row = {"window": window, "source": source}
	row["mean predicted anomaly %"] = rows["predicted_anomaly"].mean()
	row["predicted loss seasons %"] = (rows["predicted_anomaly"] < LOSS_THRESHOLD).mean() * 100
	if source == "ERA5-Land":
		row["observed loss seasons %"] = (rows["observed_anomaly"] < LOSS_THRESHOLD).mean() * 100
	summary_rows.append(row)

yield_summary = pd.DataFrame(summary_rows).set_index(["window", "source"])
summary_names = ["mean predicted anomaly %", "predicted loss seasons %"]

## Ensemble: mean of the three models, lowest and highest model
ensemble_rows = []
for window in WINDOWS:
	model_summary = yield_summary.loc[window].loc[CORRECTED_MODELS]

	for summary in ["mean", "min", "max"]:
		row = {"window": window, "source": "ensemble " + summary}
		for name in summary_names:
			if summary == "mean":
				row[name] = model_summary[name].mean()
			if summary == "min":
				row[name] = model_summary[name].min()
			if summary == "max":
				row[name] = model_summary[name].max()
		ensemble_rows.append(row)

yield_summary = pd.concat([yield_summary, pd.DataFrame(ensemble_rows).set_index(["window", "source"])]).sort_index()
print(yield_summary.round(1).to_string())

yield_summary.to_csv(RESULTS_DIR / "yield_anomaly_summary.csv")
outside_shares.to_csv(RESULTS_DIR / "extrapolation_shares.csv")
print("saved results in", RESULTS_DIR)


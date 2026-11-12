## Daily distributions of CMIP6 historical and ERA5-Land
## Requires:
##   DATA/CMIP6/<model>/historical/<variable>/*.nc   (daily tasmax, tasmin, pr)
##   DATA/ERA5-Land/<variable folders>/*.nc           (daily gridded ERA5-Land: grid, land cells, bias correction)
##   DATA/OTHERS/france_regions_2016.gpkg             (region boundaries)
##   data_processed/era5_daily_regional.csv           (scripts/02_climate_indices.py)
##
## Common period of the historical simulations and ERA5-Land: 2000-2014.
## The model years are not synchronised with the observed weather, so the sources are compared
## through the statistics of their daily distributions: mean, standard deviation, minimum,
## percentiles and maximum of the daily regional values of the growing season (October to July).
##
## Multi-model mean: the daily values of the three models are averaged cell by cell
## (same region and same day): (CNRM-ESM2-1 + EC-Earth3 + IPSL-CM6A-LR) / 3.
## The resulting series is then treated as a fourth source and compared with ERA5-Land.
##
## Bias correction: every model is interpolated on the ERA5-Land grid, then corrected cell by cell
## and month by month with quantile delta mapping (xsdba) against ERA5-Land 2000-2014,
## and only then averaged over the regions, exactly as ERA5-Land.

from pathlib import Path

import pandas as pd
import yaml
import matplotlib.pyplot as plt

from amigo_yield import read_era5, build_region_mask, compute_regional_means, read_cmip6_variable, interpolate_to_grid, train_bias_correction, apply_bias_correction, CMIP6_NAMES, compute_distribution_statistics, STATISTIC_NAMES

## Settings that change with new data, in config.yml at the root of the repository
with open("config.yml", encoding="utf-8") as config_file:
	pipeline_config = yaml.safe_load(config_file)

CMIP6_DIR = Path(pipeline_config["paths"]["cmip6_dir"])
ERA5_DIR = Path(pipeline_config["paths"]["era5_dir"])
BOUNDARIES_FILE = Path(pipeline_config["paths"]["boundaries_file"])
PROCESSED_DIR = Path("data_processed")
ERA5_DAILY_FILE = PROCESSED_DIR / "era5_daily_regional.csv"
CMIP6_HISTORICAL_FILE = PROCESSED_DIR / "cmip6_historical_regional.csv"
CMIP6_CORRECTED_FILE = PROCESSED_DIR / "cmip6_historical_corrected_regional.csv"

## Chunks of the grids for the bias correction: whole time series, blocks of cells
## (xsdba needs the whole series of every cell; smaller blocks use less memory)
CHUNKS = {"time": -1, "latitude": 24, "longitude": 36}

RESULTS_DIR = Path("results") / "cmip6"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = pipeline_config["cmip6"]["models"]
VARIABLE_NAMES = ["tmax", "tmin", "precip"]

## Common period of the historical simulations and ERA5-Land
START_DATE = "2000-01-01"
END_DATE = "2014-12-31"

## Months of the growing season
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]


## ERA5-Land daily regional means over the common period
era5_daily = pd.read_csv(ERA5_DAILY_FILE, parse_dates=["date"])
is_in_period = (era5_daily["date"] >= START_DATE) & (era5_daily["date"] <= END_DATE)
era5_daily = era5_daily[is_in_period].copy()
era5_daily["source"] = "ERA5-Land"
print("ERA5-Land:", era5_daily["date"].min().date(), "to", era5_daily["date"].max().date())


## CMIP6 historical daily regional means, raw and bias corrected, same grid and regions as ERA5-Land
## For every model and variable: interpolation on the ERA5-Land grid, bias correction cell by cell
## against ERA5-Land, then regional means of the raw and of the corrected grids.
## The results are saved to CSV files: if both files exist, they are read instead of recomputed.
## Recomputing takes about 1-2 minutes per model and variable.
if CMIP6_HISTORICAL_FILE.exists() and CMIP6_CORRECTED_FILE.exists():
	print("reading", CMIP6_HISTORICAL_FILE, "and", CMIP6_CORRECTED_FILE)
	cmip6_daily = pd.read_csv(CMIP6_HISTORICAL_FILE, parse_dates=["date"])
	cmip6_corrected_daily = pd.read_csv(CMIP6_CORRECTED_FILE, parse_dates=["date"])
else:
	## ERA5-Land daily grids of the calibration period
	era5_variables, _ = read_era5(ERA5_DIR, VARIABLE_NAMES)
	for variable in VARIABLE_NAMES:
		era5_variables[variable] = era5_variables[variable].sel(time=slice(START_DATE, END_DATE))

	## ERA5-Land grid, land cells and region mask, from the first day of tmax
	first_day = era5_variables["tmax"].isel(time=0)
	first_day = first_day.drop_vars(["time", "spatial_ref"], errors="ignore")
	latitudes = first_day["latitude"]
	longitudes = first_day["longitude"]
	land_mask = first_day.notnull()
	region_mask = build_region_mask(BOUNDARIES_FILE, longitudes, latitudes)

	raw_tables = []
	corrected_tables = []
	for model in MODELS:
		raw_variables = {}
		corrected_variables = {}
		for cmip6_name in CMIP6_NAMES:
			data = read_cmip6_variable(CMIP6_DIR, model, "historical", cmip6_name, START_DATE, END_DATE)
			variable = data.name
			interpolated = interpolate_to_grid(data, latitudes, longitudes, land_mask).chunk(CHUNKS)
			observed = era5_variables[variable].drop_vars("spatial_ref", errors="ignore").chunk(CHUNKS)

			print(model, variable)
			raw_variables[variable] = interpolated
			adjustment = train_bias_correction(observed, interpolated, variable)
			corrected_variables[variable] = apply_bias_correction(adjustment, interpolated, variable)

		raw_table = compute_regional_means(raw_variables, region_mask)
		raw_table["source"] = model
		raw_tables.append(raw_table)

		corrected_table = compute_regional_means(corrected_variables, region_mask)
		corrected_table["source"] = model + " corrected"
		corrected_tables.append(corrected_table)

	cmip6_daily = pd.concat(raw_tables, ignore_index=True)
	cmip6_corrected_daily = pd.concat(corrected_tables, ignore_index=True)
	cmip6_daily.to_csv(CMIP6_HISTORICAL_FILE, index=False)
	cmip6_corrected_daily.to_csv(CMIP6_CORRECTED_FILE, index=False)
	print("saved", CMIP6_HISTORICAL_FILE, "and", CMIP6_CORRECTED_FILE)


## Multi-model mean of the daily values: for every region and day,
## (CNRM-ESM2-1 + EC-Earth3 + IPSL-CM6A-LR) / 3
multi_model_daily = cmip6_daily.groupby(["cod_regione", "date"])[VARIABLE_NAMES].mean().reset_index()
multi_model_daily["source"] = "multi-model mean"

## Check: every region and day has the three models
models_per_day = cmip6_daily.groupby(["cod_regione", "date"])["source"].nunique()
print("days with fewer than 3 models:", int((models_per_day < 3).sum()))


## Multi-model mean of the corrected daily values: for every region and day,
## (CNRM-ESM2-1 corrected + EC-Earth3 corrected + IPSL-CM6A-LR corrected) / 3
multi_model_corrected_daily = cmip6_corrected_daily.groupby(["cod_regione", "date"])[VARIABLE_NAMES].mean().reset_index()
multi_model_corrected_daily["source"] = "multi-model mean corrected"


## All sources in one table, growing season only
## One row per source, region and day
daily = pd.concat([era5_daily, cmip6_daily, multi_model_daily, cmip6_corrected_daily, multi_model_corrected_daily], ignore_index=True)
is_in_season = daily["date"].dt.month.isin(SEASON_MONTHS)
daily = daily[is_in_season]
print(daily.groupby("source")["date"].agg(["min", "max", "count"]))


## Statistics of the daily distribution of every source
## For every source and variable: mean, standard deviation, minimum, percentiles and maximum
## of all the daily regional values (13 regions x days of the growing season 2000-2014)

statistic_rows = []
## Order of the rows: ERA5-Land, raw models, raw multi-model mean, corrected models, corrected multi-model mean
CORRECTED_MODELS = [model + " corrected" for model in MODELS]
SOURCES = ["ERA5-Land"] + MODELS + ["multi-model mean"] + CORRECTED_MODELS + ["multi-model mean corrected"]

for source in SOURCES:
	source_days = daily[daily["source"] == source]

	for variable in VARIABLE_NAMES:
		row = {"source": source, "variable": variable}
		row.update(compute_distribution_statistics(source_days[variable]))
		statistic_rows.append(row)

statistics = pd.DataFrame(statistic_rows)
print(statistics)

## Tables to read: one per variable, one row per source, one column per statistic
for variable in VARIABLE_NAMES:
	table = statistics[statistics["variable"] == variable]
	table = table.set_index("source")[STATISTIC_NAMES]
	print()
	print(variable)
	print(table.round(1).to_string())


## Differences from ERA5-Land
## Temperature: source minus ERA5-Land (degC).
## Precipitation: source minus ERA5-Land (mm per day); the low percentiles are 0 for every
## source (dry days), so a relative difference would divide by zero.
era5_statistics = statistics[statistics["source"] == "ERA5-Land"].set_index("variable")[STATISTIC_NAMES]

difference_rows = []
for _, row in statistics.iterrows():
	if row["source"] == "ERA5-Land":
		continue

	difference = {"source": row["source"], "variable": row["variable"]}
	for name in STATISTIC_NAMES:
		difference[name] = row[name] - era5_statistics.loc[row["variable"], name]
	difference_rows.append(difference)

differences = pd.DataFrame(difference_rows)

for variable in VARIABLE_NAMES:
	table = differences[differences["variable"] == variable]
	table = table.set_index("source")[STATISTIC_NAMES]
	print()
	print(variable, "minus ERA5-Land")
	print(table.round(1).to_string())


## Save the tables
statistics.to_csv(RESULTS_DIR / "distribution_statistics.csv", index=False)
differences.to_csv(RESULTS_DIR / "distribution_differences.csv", index=False)
print("saved results in", RESULTS_DIR)

## Monthly climatology over France: ERA5-Land, the three models, the multi-model mean
## and the spread of the models (shaded band between the lowest and the highest model, month by month).
## Top row: raw models. Bottom row: bias-corrected models.
## France: mean of the 13 regions on every day, then mean of every calendar month over 2000-2014.
PANEL_TITLES = {
	"tmax": "Maximum temperature (degC)",
	"tmin": "Minimum temperature (degC)",
	"precip": "Precipitation (mm per day)",
}
MODEL_COLOURS = ["#eb6834", "#1baf7a", "#eda100"]
MULTI_MODEL_COLOUR = "#2a78d6"

## France mean on every day, then monthly climatology of every source
climate_table = daily.copy()
climate_table["month"] = climate_table["date"].dt.month
france_daily = climate_table.groupby(["source", "date", "month"])[VARIABLE_NAMES].mean().reset_index()
climatology = france_daily.groupby(["source", "month"])[VARIABLE_NAMES].mean()

## The two rows: raw and corrected sources
rows = [
	{"title": "raw", "models": MODELS, "multi_model": "multi-model mean"},
	{"title": "bias corrected", "models": [model + " corrected" for model in MODELS], "multi_model": "multi-model mean corrected"},
]

positions = list(range(len(SEASON_MONTHS)))

fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
for row_number, row in enumerate(rows):

	## Spread of the models of this row: lowest and highest model in every month
	model_climatology = climatology.loc[row["models"]]
	lowest_model = model_climatology.groupby("month").min()
	highest_model = model_climatology.groupby("month").max()

	for column_number, variable in enumerate(VARIABLE_NAMES):
		ax = axes[row_number, column_number]

		## Spread of the models
		ax.fill_between(positions, lowest_model.loc[SEASON_MONTHS, variable], highest_model.loc[SEASON_MONTHS, variable], color=MULTI_MODEL_COLOUR, alpha=0.15, linewidth=0, label="spread of the models")

		## Every model, thin dashed lines
		for model, colour in zip(row["models"], MODEL_COLOURS):
			ax.plot(positions, climatology.loc[(model, SEASON_MONTHS), variable].values, color=colour, linewidth=1.2, linestyle="--", label=model.replace(" corrected", ""))

		## Multi-model mean
		ax.plot(positions, climatology.loc[(row["multi_model"], SEASON_MONTHS), variable].values, color=MULTI_MODEL_COLOUR, linewidth=2.5, label="multi-model mean")

		## ERA5-Land
		ax.plot(positions, climatology.loc[("ERA5-Land", SEASON_MONTHS), variable].values, color="black", linewidth=2.5, label="ERA5-Land")

		ax.set_xticks(positions)
		ax.set_xticklabels(MONTH_LABELS, fontsize=11)
		ax.tick_params(axis="y", labelsize=11)
		ax.set_title(PANEL_TITLES[variable] + ", " + row["title"], fontsize=13)
		ax.grid(alpha=0.3)

## One legend for the whole figure, below the panels
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.suptitle("Monthly climatology over France, 2000-2014: mean of all the days of each month over the 15 years", fontsize=14)
fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=11, frameon=False)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(RESULTS_DIR / "ensemble_seasonal_cycle.png", dpi=150)

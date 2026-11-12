# # Exploration of the climate data
## Requires data_processed/yield_anomalies.csv and national_yield_anomaly.csv:
## run scripts/01_yield_anomalies.py first.

# %% Read all ERA5-Land variables
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from amigo_yield import read_era5, compute_wind_speed, build_region_mask, compute_regional_means, add_harvest_year

## Settings that change with new data, in config.yml at the root of the repository
with open("config.yml", encoding="utf-8") as config_file:
	pipeline_config = yaml.safe_load(config_file)

ERA5_DIR = Path(pipeline_config["paths"]["era5_dir"])
BOUNDARIES_FILE = Path(pipeline_config["paths"]["boundaries_file"])
PROCESSED_DIR = Path("data_processed")

era5_variables, era5_summary = read_era5(ERA5_DIR)
print(era5_summary.to_string(index=False))


# %% Daily wind speed from the u and v components
era5_variables["wind"] = compute_wind_speed(era5_variables["u10"], era5_variables["v10"])


# %% Monthly means on the grid
## Daily data are reduced to monthly means first: everything after this cell is fast
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]   # October to July, no August and September

monthly_variables = {}
for short_name, data in era5_variables.items():
	print("monthly means of", short_name)
	monthly = data.resample(time="MS").mean()
	monthly_variables[short_name] = monthly.compute()


# %% Simple mean over the grid cells of each region
## A cell belongs to a region if its centre falls inside the polygon
first_variable = list(monthly_variables.values())[0]
region_mask = build_region_mask(BOUNDARIES_FILE, first_variable["longitude"], first_variable["latitude"])

## Number of cells in each region (must be greater than zero)
cells_per_region = region_mask.sum(dim=["latitude", "longitude"])
print(cells_per_region.to_series())

monthly_climate = compute_regional_means(monthly_variables, region_mask)

## Keep October to July and label every month with its harvest year
monthly_climate = add_harvest_year(monthly_climate)

# %% Join the monthly climate with the yield anomalies
yield_anomalies = pd.read_csv(PROCESSED_DIR / "yield_anomalies.csv")
yield_anomalies = yield_anomalies.rename(columns={"year": "harvest_year"})

variable_names = list(monthly_variables.keys())

## Keep only the months where every variable has data.
## Precipitation covers 1991-2022, the other variables 2000-2019:
## without this step the months with precipitation only would count as complete.
has_all_variables = monthly_climate[variable_names].notna().all(axis=1)
monthly_complete = monthly_climate[has_all_variables]

monthly_with_yield = monthly_complete.merge(yield_anomalies, on=["cod_regione", "harvest_year"], how="inner")

## Keep only seasons with all ten months
months_per_season = monthly_with_yield.groupby(["cod_regione", "harvest_year"])["month"].transform("count")
is_complete_season = months_per_season == len(SEASON_MONTHS)
monthly_with_yield = monthly_with_yield[is_complete_season]

print("harvest years:", monthly_with_yield["harvest_year"].min(), "to", monthly_with_yield["harvest_year"].max())
print("region-years:", monthly_with_yield.groupby(["cod_regione", "harvest_year"]).ngroups)


# %% Climate of the loss years, from the worst to the least bad
## Season mean (October to July) of every variable for every region and harvest year
LOSS_THRESHOLD = -10   # yield anomaly in %

season_climate = monthly_with_yield.groupby(["cod_regione", "regione", "harvest_year"])[variable_names].mean()
season_climate = season_climate.reset_index()

## Add the yield anomaly of each region and year
season_climate = season_climate.merge(
	yield_anomalies[["cod_regione", "harvest_year", "anomaly_pct"]],
	on=["cod_regione", "harvest_year"],
	how="inner",
)

## Normal value of each variable in each region: mean over all seasons 2001-2019
region_normals = season_climate.groupby("cod_regione")[variable_names].mean()

## Difference from the normal of the same region (positive = above normal)
season_difference = season_climate.copy()
for variable in variable_names:
	normal_values = season_difference["cod_regione"].map(region_normals[variable])
	season_difference[variable] = season_difference[variable] - normal_values

## Region-years with a loss larger than the threshold, from the worst
is_loss = season_climate["anomaly_pct"] < LOSS_THRESHOLD
loss_values = season_climate[is_loss].sort_values("anomaly_pct")
loss_differences = season_difference[is_loss].sort_values("anomaly_pct")

print("region-years with a loss below", LOSS_THRESHOLD, "%:", len(loss_values))
print()
print("Season mean values (October to July):")
print(loss_values.round(2).to_string(index=False))
print()
print("Difference from the normal of the region:")
print(loss_differences.round(2).to_string(index=False))


# %% Function: monthly climate and yield anomaly over the whole period
## monthly_table: one row per month with the columns date, month, harvest_year,
## the climate variables and anomaly_pct (one value per season).
## Panels share the time axis: one per climate variable (difference from the normal
## of that month) and one with the yield anomaly. Loss seasons are shaded in grey.
PLOT_VARIABLES = ["tmax", "tmin", "precip", "wind"]

def plot_climate_and_yield(monthly_table, title, variables, loss_threshold):
	table = monthly_table.sort_values("date").copy()

	## Difference from the normal of the same month
	for variable in variables:
		month_normal = table.groupby("month")[variable].transform("mean")
		table[variable + "_diff"] = table[variable] - month_normal

	## One yield anomaly per season, drawn as a bar from October to July
	seasons = table.groupby("harvest_year")["anomaly_pct"].first()
	season_starts = pd.to_datetime([str(year - 1) + "-10-01" for year in seasons.index])
	season_ends = pd.to_datetime([str(year) + "-07-31" for year in seasons.index])

	number_of_panels = len(variables) + 1
	fig, axes = plt.subplots(number_of_panels, 1, figsize=(14, 2.4 * number_of_panels), sharex=True)

	## Climate panels: red above normal, blue below normal
	for ax, variable in zip(axes, variables):
		values = table[variable + "_diff"]

		colours = []
		for value in values:
			if value > 0:
				colours.append("#b2302f")
			else:
				colours.append("#1c5cab")

		ax.bar(table["date"], values, width=25, color=colours)
		ax.axhline(0, color="0.4", lw=0.8)
		units = era5_variables[variable].attrs.get("units", "")
		ax.set_ylabel(variable + " diff\n(" + units + ")", fontsize=9)
		ax.grid(axis="y", alpha=0.3)

	## Yield panel
	yield_ax = axes[-1]
	for start, end, anomaly in zip(season_starts, season_ends, seasons.values):
		if anomaly < 0:
			colour = "#b2302f"
		else:
			colour = "#1c5cab"
		yield_ax.bar(start, anomaly, width=(end - start).days, align="edge", color=colour, alpha=0.8)

	yield_ax.axhline(0, color="0.4", lw=0.8)
	yield_ax.set_ylabel("yield anomaly\n(%)", fontsize=9)
	yield_ax.grid(axis="y", alpha=0.3)

	## Shade the loss seasons in every panel
	for start, end, anomaly in zip(season_starts, season_ends, seasons.values):
		if anomaly < loss_threshold:
			for ax in axes:
				ax.axvspan(start, end, color="0.85", zorder=0)

	axes[0].set_title(title)
	fig.tight_layout()
	plt.show()


# %% All of France
## Yield: national anomaly computed in the yield script (total production / total area)
## Climate: mean of the regions for every month
national_anomaly = pd.read_csv(PROCESSED_DIR / "national_yield_anomaly.csv")

france_climate = monthly_with_yield.groupby(["date", "month", "harvest_year"])[PLOT_VARIABLES].mean()
france_climate = france_climate.reset_index()

france_monthly = france_climate.merge(national_anomaly, on="harvest_year", how="inner")

plot_climate_and_yield(
	france_monthly,
	"France: monthly climate (difference from normal) and national soft wheat yield anomaly",
	PLOT_VARIABLES,
	LOSS_THRESHOLD,
)


# %% Ranking of the regions by their worst season
## Worst season = lowest yield anomaly of the region among the seasons with climate data (2001-2019)
worst_season = monthly_with_yield.groupby(["cod_regione", "regione"])["anomaly_pct"].min()
worst_season = worst_season.sort_values()
print("worst season of each region (%):")
print(worst_season.round(1).to_string())


# %% Region with the largest loss
## First of the ranking: the region whose worst season was the most severe
largest_loss = worst_season.index[0]
largest_loss_code = largest_loss[0]
largest_loss_name = largest_loss[1]
largest_loss_value = worst_season.iloc[0]

is_region = monthly_with_yield["cod_regione"] == largest_loss_code
largest_loss_monthly = monthly_with_yield[is_region]

title = largest_loss_name + " (largest loss, worst season " + str(round(largest_loss_value)) + "%): monthly climate and yield anomaly"
plot_climate_and_yield(largest_loss_monthly, title, PLOT_VARIABLES, LOSS_THRESHOLD)


# %% Region with the smallest loss
## Last of the ranking: the region whose worst season was the least severe
smallest_loss = worst_season.index[-1]
smallest_loss_code = smallest_loss[0]
smallest_loss_name = smallest_loss[1]
smallest_loss_value = worst_season.iloc[-1]

is_region = monthly_with_yield["cod_regione"] == smallest_loss_code
smallest_loss_monthly = monthly_with_yield[is_region]

title = smallest_loss_name + " (smallest loss, worst season " + str(round(smallest_loss_value)) + "%): monthly climate and yield anomaly"
plot_climate_and_yield(smallest_loss_monthly, title, PLOT_VARIABLES, LOSS_THRESHOLD)
# %%Wind and precipitation of every region-season in two windows, as deviations from the
## regional mean, compared with the yield anomaly.
WINDOWS = {
	"oct_feb": [10, 11, 12, 1, 2],
	"may_jul": [5, 6, 7],
}

result_rows = []
for window_name, months in WINDOWS.items():

	## Mean of every region-season over the months of the window
	window_rows = monthly_with_yield[monthly_with_yield["month"].isin(months)]
	seasonal = window_rows.groupby(["cod_regione", "harvest_year"])[["wind", "precip", "anomaly_pct"]].mean()

	## Deviation of wind and precipitation from the mean of their region
	regional_means = seasonal.groupby("cod_regione")[["wind", "precip"]].transform("mean")
	wind_deviation = seasonal["wind"] - regional_means["wind"]
	precip_deviation = seasonal["precip"] - regional_means["precip"]
	yield_anomaly = seasonal["anomaly_pct"]

	## 1. Redundancy: correlation between wind and precipitation
	wind_precip = np.corrcoef(wind_deviation, precip_deviation)[0, 1]

	## 2. Correlation between wind and yield
	wind_yield = np.corrcoef(wind_deviation, yield_anomaly)[0, 1]

	## 3. Partial correlation: remove from wind and yield the part explained by precipitation
	wind_slope, wind_intercept = np.polyfit(precip_deviation, wind_deviation, deg=1)
	wind_residual = wind_deviation - (wind_slope * precip_deviation + wind_intercept)

	yield_slope, yield_intercept = np.polyfit(precip_deviation, yield_anomaly, deg=1)
	yield_residual = yield_anomaly - (yield_slope * precip_deviation + yield_intercept)

	wind_yield_partial = np.corrcoef(wind_residual, yield_residual)[0, 1]

	row = {
		"window": window_name,
		"wind_vs_precip": wind_precip,
		"wind_vs_yield": wind_yield,
		"wind_vs_yield_without_precip": wind_yield_partial,
	}
	result_rows.append(row)

print(pd.DataFrame(result_rows).round(2).to_string(index=False))
# %% Correlation between monthly climate and yield anomaly
## For every month of the season and every variable: correlation between the deviation
## from the regional normal of that month and the yield anomaly of the season.
CORRELATION_VARIABLES = ["tmax", "tmin", "precip", "wind"]
SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]
MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]

## Normal of every region and month, and deviation of every region-season from it
monthly_normals = monthly_with_yield.groupby(["cod_regione", "month"])[CORRELATION_VARIABLES].transform("mean")
deviations = monthly_with_yield[CORRELATION_VARIABLES] - monthly_normals
deviations["month"] = monthly_with_yield["month"]
deviations["anomaly_pct"] = monthly_with_yield["anomaly_pct"]

## One correlation for every month and variable
correlations = pd.DataFrame(index=MONTH_LABELS, columns=CORRELATION_VARIABLES, dtype=float)
for month, label in zip(SEASON_MONTHS, MONTH_LABELS):
	month_rows = deviations[deviations["month"] == month]
	for variable in CORRELATION_VARIABLES:
		correlations.loc[label, variable] = np.corrcoef(month_rows[variable], month_rows["anomaly_pct"])[0, 1]

print(correlations.round(2).to_string())

## Heat map: red = more of the variable goes with lower yields, blue = with higher yields
fig, ax = plt.subplots(figsize=(5, 6))
image = ax.imshow(correlations.to_numpy(), cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks(range(len(CORRELATION_VARIABLES)))
ax.set_xticklabels(CORRELATION_VARIABLES)
ax.set_yticks(range(len(MONTH_LABELS)))
ax.set_yticklabels(MONTH_LABELS)
for row in range(len(MONTH_LABELS)):
	for column in range(len(CORRELATION_VARIABLES)):
		ax.text(column, row, f"{correlations.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(image, ax=ax, label="correlation with yield anomaly")
ax.set_title("Monthly climate deviation vs yield anomaly")
fig.tight_layout()
plt.show()


# %% Correlation between the variables, month by month
## High values = two variables carry largely the same information in that month
pairs = [("tmax", "tmin"), ("tmax", "precip"), ("precip", "wind"), ("tmin", "precip")]

pair_correlations = pd.DataFrame(index=MONTH_LABELS, dtype=float)
for first, second in pairs:
	pair_name = first + " vs " + second
	for month, label in zip(SEASON_MONTHS, MONTH_LABELS):
		month_rows = deviations[deviations["month"] == month]
		pair_correlations.loc[label, pair_name] = np.corrcoef(month_rows[first], month_rows[second])[0, 1]

print(pair_correlations.round(2).to_string())
# %% November wind: effect of precipitation, or of a single season?
november = deviations[deviations["month"] == 11].copy()
november["harvest_year"] = monthly_with_yield.loc[november.index, "harvest_year"]

## 1. Partial correlation: remove from wind and yield the part explained by precipitation
wind_slope, wind_intercept = np.polyfit(november["precip"], november["wind"], deg=1)
wind_residual = november["wind"] - (wind_slope * november["precip"] + wind_intercept)
yield_slope, yield_intercept = np.polyfit(november["precip"], november["anomaly_pct"], deg=1)
yield_residual = november["anomaly_pct"] - (yield_slope * november["precip"] + yield_intercept)
print("wind vs yield:", round(np.corrcoef(november["wind"], november["anomaly_pct"])[0, 1], 2))
print("wind vs yield without precipitation:", round(np.corrcoef(wind_residual, yield_residual)[0, 1], 2))

## 2. The same correlation without 2016
without_2016 = november[november["harvest_year"] != 2016]
print("wind vs yield without 2016:", round(np.corrcoef(without_2016["wind"], without_2016["anomaly_pct"])[0, 1], 2))

## 3. Scatter: every point is a region-season, the 2016 season in red
fig, ax = plt.subplots(figsize=(5, 4))
is_2016 = november["harvest_year"] == 2016
ax.scatter(november.loc[~is_2016, "wind"], november.loc[~is_2016, "anomaly_pct"], s=14, color="0.5", label="other seasons")
ax.scatter(november.loc[is_2016, "wind"], november.loc[is_2016, "anomaly_pct"], s=20, color="#b2302f", label="2016")
ax.set_xlabel("November wind, deviation from regional normal (m/s)")
ax.set_ylabel("yield anomaly (%)")
ax.legend()
fig.tight_layout()
plt.show()
# %% November wind in loss seasons and in the other seasons
is_loss = november["anomaly_pct"] < -10

print("loss seasons:", is_loss.sum(), "| mean November wind deviation:", round(november.loc[is_loss, "wind"].mean(), 2))
print("other seasons:", (~is_loss).sum(), "| mean November wind deviation:", round(november.loc[~is_loss, "wind"].mean(), 2))

## Share of loss seasons among the windy and the calm Novembers
is_windy = november["wind"] > 0
print("losses after a windy November:", round(is_loss[is_windy].mean(), 2))
print("losses after a calm November:", round(is_loss[~is_windy].mean(), 2))
# %%
# %% Correlation between the climate variables (deviations from the regional monthly normal)
# %%
# %% In how many different years do the losses after a windy November fall?
losses_after_windy = november[is_loss & is_windy]
print(losses_after_windy.groupby("harvest_year").size().to_string())
# %%

# %% Read all ERA5-Land variables
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ERA5_DIR = Path("DATA/ERA5-Land")
BOUNDARIES_FILE = Path("DATA/OTHERS/france_regions_2016.gpkg")
PROCESSED_DIR = Path("data_processed")

FOLDER_KEYWORDS = {
	"maximum": "tmax",
	"minimum": "tmin",
	"precipitation": "precip",
	"u_wind": "u10",
	"v_wind": "v10",
	"pressure": "sp",
}


## Short name of a variable from the name of its folder
def get_short_name(folder_name):
	lower_name = folder_name.lower()
	for keyword, short_name in FOLDER_KEYWORDS.items():
		if keyword in lower_name:
			return short_name
	return lower_name


## Convert to common units: K -> degC, m of water -> mm, Pa -> hPa
def convert_units(data):
	units = data.attrs.get("units", "")

	if units == "K":
		data = data - 273.15
		data.attrs["units"] = "degC"
	elif units == "m":
		data = data * 1000
		data.attrs["units"] = "mm"
	elif units == "Pa":
		data = data / 100
		data.attrs["units"] = "hPa"

	return data


## Read every variable folder of ERA5-Land.
## Returns a dictionary short name -> data array (time x latitude x longitude)
## and a table with the time frame and resolution of each variable.
def read_era5(era5_dir):
	variables = {}
	summary_rows = []
	variable_folders = []
	for path in era5_dir.iterdir():
		if path.is_dir():
			variable_folders.append(path)
	variable_folders = sorted(variable_folders)	
	for variable_folder in variable_folders:
		short_name = get_short_name(variable_folder.name)
		monthly_files = sorted(variable_folder.glob("*.nc"))

		dataset = xr.open_mfdataset(monthly_files, combine="by_coords", data_vars="minimal", coords="minimal", compat="override")

		## Keep only the data variable, without extra coordinates 
		variable_name = []

		for name in dataset.data_vars:
			if name != "spatial_ref":
				variable_name.append(name)

		variable_name = variable_name[0]
		data = dataset[variable_name]
		data = data.drop_vars(["expver", "number"], errors="ignore")
		data = convert_units(data)
		rounded_latitude = np.round(data["latitude"].values, 2)
		rounded_longitude = np.round(data["longitude"].values, 2)
		data = data.assign_coords(latitude=rounded_latitude, longitude=rounded_longitude)
		data = data.sortby("latitude")

		times = data["time"].values
		row = {
			"variable": short_name,
			"folder": variable_folder.name,
			"units": data.attrs.get("units"),
			"first_day": str(times[0])[:10],
			"last_day": str(times[-1])[:10],
			"n_days": len(times),
			"lat_step": round(float(np.median(np.diff(data["latitude"].values))), 3),
			"lon_step": round(float(np.median(np.diff(data["longitude"].values))), 3),
			"grid": str(data.sizes["latitude"]) + " x " + str(data.sizes["longitude"]),
		}
		summary_rows.append(row)
		variables[short_name] = data

	summary = pd.DataFrame(summary_rows)
	return variables, summary


era5_variables, era5_summary = read_era5(ERA5_DIR)
print(era5_summary.to_string(index=False))


# %% Daily wind speed from the u and v components
## Speed = sqrt(u^2 + v^2), computed day by day before any averaging.
## The daily u and v are already daily means, so this speed is slightly lower
## than the true mean speed (winds from opposite directions within a day cancel out).
u_wind = era5_variables["u10"]
v_wind = era5_variables["v10"]

wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
wind_speed.attrs["units"] = "m/s"

era5_variables["wind"] = wind_speed


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
regions = gpd.read_file(BOUNDARIES_FILE)

region_masks = regionmask.from_geopandas(regions, numbers="cod_regione", names="regione", name="regions")
first_variable = list(monthly_variables.values())[0]
region_mask = region_masks.mask_3D(first_variable["longitude"], first_variable["latitude"])

## Number of cells in each region (must be greater than zero)
cells_per_region = region_mask.sum(dim=["latitude", "longitude"])
print(cells_per_region.to_series())

monthly_tables = []
for short_name, monthly in monthly_variables.items():
	regional = monthly.where(region_mask).mean(dim=["latitude", "longitude"])
	table = regional.to_dataframe(name=short_name)[[short_name]]
	monthly_tables.append(table)

monthly_climate = pd.concat(monthly_tables, axis=1)
monthly_climate = monthly_climate.reset_index()
monthly_climate = monthly_climate.rename(columns={"region": "cod_regione", "time": "date"})

## Keep October to July and label every month with its harvest year
monthly_climate["month"] = monthly_climate["date"].dt.month
monthly_climate["harvest_year"] = monthly_climate["date"].dt.year
is_autumn = monthly_climate["month"] >= 10
monthly_climate.loc[is_autumn, "harvest_year"] = monthly_climate.loc[is_autumn, "harvest_year"] + 1
monthly_climate = monthly_climate[monthly_climate["month"].isin(SEASON_MONTHS)]

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
# %%

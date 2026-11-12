## ERA5-Land reading, regional means and harvest year

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

## Short names of the variables, found from keywords in the folder names
## (the files of tmax and tmin both call the variable t2m)
FOLDER_KEYWORDS = {
	"maximum": "tmax",
	"minimum": "tmin",
	"precipitation": "precip",
	"u_wind": "u10",
	"v_wind": "v10",
	"pressure": "sp",
}

## Growing season of winter soft wheat: sowing in October of year t-1, harvest in July of year t
SEASON_START_MONTH = 10
SEASON_END_MONTH = 7
SECONDS_PER_DAY = 86400

## Percentiles of the daily distributions, and names of all the statistics in the order of the tables
PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
STATISTIC_NAMES = ["mean", "std", "min"] + ["p" + str(p) for p in PERCENTILES] + ["max"]


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
	elif units == "kg m-2 s-1":
		data = data * SECONDS_PER_DAY
		data.attrs["units"] = "mm"
	return data


## Read the ERA5-Land variable folders (one folder per variable, one file per month).
## variable_names: short names to read (for example ["tmax", "tmin", "precip"]); None reads every folder.
## Returns a dictionary short name -> daily data (time x latitude x longitude)
## and a table with the time frame and resolution of each variable.
def read_era5(era5_dir, variable_names=None):
	variables = {}
	summary_rows = []

	variable_folders = []
	for path in era5_dir.iterdir():
		if path.is_dir():
			variable_folders.append(path)
	variable_folders = sorted(variable_folders)

	for variable_folder in variable_folders:
		short_name = get_short_name(variable_folder.name)
		if variable_names is not None and short_name not in variable_names:
			continue

		monthly_files = sorted(variable_folder.glob("*.nc"))
		dataset = xr.open_mfdataset(monthly_files, combine="by_coords", data_vars="minimal", coords="minimal", compat="override")

		## Keep only the data variable, without extra coordinates
		data_variable_names = []
		for name in dataset.data_vars:
			if name != "spatial_ref":
				data_variable_names.append(name)

		data = dataset[data_variable_names[0]]
		data = data.drop_vars(["expver", "number"], errors="ignore")
		data = convert_units(data)

		## Same grid for every variable: rounded coordinates, latitude from south to north.
		## Some folders store latitude in decreasing order, and float32 coordinates
		## can differ in the last decimals between files.
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


## Daily wind speed from the u and v components: sqrt(u^2 + v^2), day by day before any averaging.
## The daily u and v are already daily means, so this speed is slightly lower
## than the true mean speed (winds from opposite directions within a day cancel out).
def compute_wind_speed(u_wind, v_wind):
	wind_speed = np.sqrt(u_wind ** 2 + v_wind ** 2)
	wind_speed.attrs["units"] = "m/s"
	return wind_speed


## Mask of the regions on the grid: region x latitude x longitude, True inside the region.
## A cell belongs to a region if its centre falls inside the polygon (no weights).
def build_region_mask(boundaries_file, longitudes, latitudes):
	regions = gpd.read_file(boundaries_file)
	region_masks = regionmask.from_geopandas(regions, numbers="cod_regione", names="regione", name="regions")
	region_mask = region_masks.mask_3D(longitudes, latitudes)
	return region_mask


## Mean of every variable over each region, for every time step (daily or monthly data).
## Sum of the values inside each region, divided by the number of cells with data:
## sea cells (NaN in ERA5-Land) count as zero in the sum and are not counted.
## Returns a table with the columns cod_regione, date and one column per variable.
def compute_regional_means(variables, region_mask):
	## Mask as numbers (1 inside the region, 0 outside) for the sums
	mask = region_mask.astype("float32")
	tables = []

	for short_name, data in variables.items():
		print("regional means of", short_name)

		values_sum = xr.dot(data.fillna(0), mask, dim=["latitude", "longitude"])
		cells_with_data = xr.dot(data.notnull().astype("float32"), mask, dim=["latitude", "longitude"])
		regional = values_sum / cells_with_data
		regional = regional.compute()

		table = regional.to_dataframe(name=short_name)[[short_name]]
		tables.append(table)

	regional_table = pd.concat(tables, axis=1)
	regional_table = regional_table.reset_index()
	regional_table = regional_table.rename(columns={"region": "cod_regione", "time": "date"})
	return regional_table


## Add the columns month and harvest_year, and keep only the months of the growing season.
## October to December belong to the harvest of the following year; August and September are removed.
def add_harvest_year(table, date_column="date"):
	table = table.copy()
	table["month"] = table[date_column].dt.month
	table["harvest_year"] = table[date_column].dt.year

	is_autumn = table["month"] >= SEASON_START_MONTH
	table.loc[is_autumn, "harvest_year"] = table.loc[is_autumn, "harvest_year"] + 1

	is_in_season = (table["month"] >= SEASON_START_MONTH) | (table["month"] <= SEASON_END_MONTH)
	table = table[is_in_season]
	return table


## Statistics of a daily distribution: mean, standard deviation, minimum, percentiles and maximum
## of all the values, missing values removed.
## Returns a dictionary with one entry for every name in STATISTIC_NAMES.
def compute_distribution_statistics(values):
	values = np.asarray(values, dtype=float)
	values = values[~np.isnan(values)]

	statistics = {}
	statistics["mean"] = values.mean()
	statistics["std"] = values.std()
	statistics["min"] = values.min()
	for p in PERCENTILES:
		statistics["p" + str(p)] = np.percentile(values, p)
	statistics["max"] = values.max()
	return statistics

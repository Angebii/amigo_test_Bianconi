## CMIP6 daily data: reading, interpolation on the ERA5-Land grid and bias correction.
## The models go through the same unit conversion (convert_units), the same grid and the same
## regional means (compute_regional_means) as ERA5-Land, so the two are compared after
## exactly the same processing.

from pathlib import Path

import xarray as xr
from xsdba import QuantileDeltaMapping

from amigo_yield.climate import convert_units


## CMIP6 variable names and the short names used for ERA5-Land
CMIP6_NAMES = {
	"tasmax": "tmax",
	"tasmin": "tmin",
	"pr": "precip",
}


## Daily data of one CMIP6 variable for one model and one experiment, between two dates.
## cmip6_dir: folder with one subfolder per model, then per experiment, then per variable
## Returns a DataArray (time, latitude, longitude) in degC or mm per day,
## named with the ERA5-Land short name.
def read_cmip6_variable(cmip6_dir, model, experiment, variable, start_date, end_date):
	folder = Path(cmip6_dir) / model / experiment / variable
	files = sorted(folder.glob("*.nc"))
	if len(files) == 0:
		raise FileNotFoundError(f"no file in {folder}")

	dataset = xr.open_mfdataset(files, combine="by_coords")
	data = dataset[variable].sel(time=slice(start_date, end_date))

	## Same coordinate names and same order of latitude as ERA5-Land
	data = data.rename({"lat": "latitude", "lon": "longitude"})
	data = data.sortby("latitude")

	## Units from the attributes of the file: K -> degC, kg m-2 s-1 -> mm per day
	data = convert_units(data)

	## Daily values: keep only the date (some models store the time at noon)
	data["time"] = data["time"].dt.floor("D")

	return data.rename(CMIP6_NAMES[variable])


## Bilinear interpolation of a CMIP6 field on the ERA5-Land grid.
## land_mask: True on the ERA5-Land cells with data; cells over the sea are set to NaN,
## as in ERA5-Land, so that the regional means use exactly the same cells.
def interpolate_to_grid(data, latitudes, longitudes, land_mask):
	interpolated = data.interp(latitude=latitudes, longitude=longitudes, method="linear")
	interpolated = interpolated.where(land_mask)
	return interpolated


## Precipitation below this value (mm per day) counts as a dry day: 0.05 mm, the default trace
## of the reference implementation of quantile delta mapping (R package MBC, function QDM, by A. J. Cannon)
PRECIPITATION_TRACE = 0.05

## Number of quantiles of the empirical distributions in every grid cell and calendar month
NUMBER_OF_QUANTILES = 50

## Kind of correction and units of every variable, as xsdba wants them:
## temperature additive, precipitation multiplicative
VARIABLE_KINDS = {"tmax": "+", "tmin": "+", "precip": "*"}
VARIABLE_UNITS = {"tmax": "degC", "tmin": "degC", "precip": "mm/d"}


## Bias correction of a gridded CMIP6 variable with quantile delta mapping (Cannon et al., 2015),
## implementation of the xsdba package (statistical downscaling and bias adjustment of xclim,
## now a separate package). Every grid cell and calendar month gets its own correction.
## observed: ERA5-Land of the calibration period (time, latitude, longitude)
## model_historical: model of the calibration period, interpolated on the same grid
## model_target: model values to correct (the calibration period itself or a future period), same grid
## variable: short name (tmax, tmin, precip)
## Precipitation: values below the trace are replaced by small random values before the
## correction (jitter_under_thresh), and corrected values below the trace are set back to zero.
## Sea cells (NaN in ERA5-Land and in the interpolated model) stay NaN.
## The arrays should be chunked in space and not in time (xsdba needs the whole series of every cell).
def bias_correct_grid(observed, model_historical, model_target, variable):
	kind = VARIABLE_KINDS[variable]
	units = VARIABLE_UNITS[variable]

	observed = observed.assign_attrs(units=units)
	model_historical = model_historical.assign_attrs(units=units)
	model_target = model_target.assign_attrs(units=units)

	if kind == "+":
		adjustment = QuantileDeltaMapping.train(
			observed, model_historical,
			nquantiles=NUMBER_OF_QUANTILES, kind="+", group="time.month",
		)
	else:
		adjustment = QuantileDeltaMapping.train(
			observed, model_historical,
			nquantiles=NUMBER_OF_QUANTILES, kind="*", group="time.month",
			jitter_under_thresh_value=f"{PRECIPITATION_TRACE} {units}",
		)

	corrected = adjustment.adjust(model_target, interp="linear")

	## xsdba may return the dimensions in another order: back to the order of the input
	corrected = corrected.transpose(*model_target.dims)

	if kind == "*":
		is_dry = corrected < PRECIPITATION_TRACE
		corrected = corrected.where(~is_dry, 0.0)

	return corrected.rename(variable)
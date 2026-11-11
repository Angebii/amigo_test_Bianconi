## Climate indices over windows of months

import pandas as pd

from amigo_yield.climate import add_harvest_year


## One statistic of one variable in one window of months, for every region and harvest year.
## daily: table with the columns cod_regione, harvest_year, month and the variable.
## statistic: "sum", "mean" or "days_above" (number of days with value above the threshold)
## Windows with missing days are set to NaN (for example the first season of the data).
def compute_window_index(daily, variable, months, statistic, threshold=None):
	window = daily[daily["month"].isin(months)]
	grouped = window.groupby(["cod_regione", "harvest_year"])[variable]

	if statistic == "sum":
		index = grouped.sum()
	elif statistic == "mean":
		index = grouped.mean()
	elif statistic == "days_above":
		is_above = window[variable] > threshold
		index = is_above.groupby([window["cod_regione"], window["harvest_year"]]).sum()
	else:
		raise ValueError("Unknown statistic: " + statistic)

	## A complete window has at least 28 days for each month
	number_of_days = grouped.count()
	is_complete = number_of_days >= 28 * len(months)
	index = index.where(is_complete)

	return index


## Threshold of a hot day: maximum temperature above 30 degC
HOT_DAY_THRESHOLD = 30

## Names of the indices of the growing season, in the order used by the Ridge regression
INDEX_NAMES = [
	"precip_oct_jan",
	"tmean_nov_dec",
	"precip_mar_apr",
	"precip_may_jun",
	"temperature_range_may_jun",
	"tmax_may_jun",
	"hot_days_may_jul",
]


## The seven indices of the growing season for every region and harvest year, from daily regional data
## (ERA5-Land or a bias-corrected climate model). Windows and thresholds are set here, in one place.
## daily: table with the columns cod_regione, date, tmax, tmin and precip
## Returns a table with the columns cod_regione, harvest_year and one column per index.
def compute_season_indices(daily):

	## Keep only the days where all the variables have data
	has_all_variables = daily[["tmax", "tmin", "precip"]].notna().all(axis=1)
	daily = daily[has_all_variables]

	## October to December belong to the harvest of the following year; August and September are removed
	daily = add_harvest_year(daily)

	## Daily mean temperature and daily temperature range
	daily["tmean"] = (daily["tmax"] + daily["tmin"]) / 2
	daily["temperature_range"] = daily["tmax"] - daily["tmin"]

	indices = {}

	## 1. Autumn-winter rain: waterlogged soils at sowing and during winter (2001; west of France)
	indices["precip_oct_jan"] = compute_window_index(daily, "precip", [10, 11, 12, 1], "sum")

	## 2. Late autumn temperature: warm late autumn, first part of the 2016 compound event (Ben-Ari et al. 2018)
	indices["tmean_nov_dec"] = compute_window_index(daily, "tmean", [11, 12], "mean")

	## 3. Early spring rain: spring drought (2003, 2011); rain in March-April also favours foliar diseases
	indices["precip_mar_apr"] = compute_window_index(daily, "precip", [3, 4], "sum")

	## 4. Late spring rain around anthesis: wet spring, second part of the 2016 compound event
	indices["precip_may_jun"] = compute_window_index(daily, "precip", [5, 6], "sum")

	## 5. Temperature range in late spring: indirect signal of cloud cover (no radiation in the data)
	indices["temperature_range_may_jun"] = compute_window_index(daily, "temperature_range", [5, 6], "mean")

	## 6. Late spring heat: mean tmax during anthesis and grain filling (2003; Gammans et al. 2017)
	indices["tmax_may_jun"] = compute_window_index(daily, "tmax", [5, 6], "mean")

	## 7. Hot days during anthesis and grain filling: days with tmax above 30 degC (Noia Junior et al. 2025)
	indices["hot_days_may_jul"] = compute_window_index(daily, "tmax", [5, 6, 7], "days_above", HOT_DAY_THRESHOLD)

	index_table = pd.DataFrame(indices)
	index_table = index_table.reset_index()
	return index_table

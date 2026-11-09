## Climate indices over windows of months

import pandas as pd


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

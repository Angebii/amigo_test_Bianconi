## Agreste yield tables and regional yield anomalies

import numpy as np
import pandas as pd


## Build a table with regions on rows and years on columns,
## for one crop code and one indicator (Produzione, Superficie, Resa (calcolata))
def get_crop_table(table, crop_code, indicator):
	is_crop = table["codice"] == crop_code
	is_indicator = table["indicatore"] == indicator
	selection = table[is_crop & is_indicator]

	crop_table = selection.pivot(index="regione", columns="anno", values="valore")
	return crop_table


## For every region: fit a linear trend on all the years of the table and compute the relative anomaly (%).
## Returns the anomaly table (regions x years) and the trend slope of every region.
## The trend uses every year in the table: fine for exploration only.
## For model evaluation use fit_trends and compute_anomalies in baseline.py, on training years only.
def compute_regional_anomalies(yield_table):
	years = yield_table.columns.to_numpy()

	anomaly_dict = {}
	slope_dict = {}

	for region, region_yields in yield_table.iterrows():
		values = region_yields.to_numpy()

		slope, intercept = np.polyfit(years, values, deg=1)
		expected_yield = slope * years + intercept

		anomaly = (values - expected_yield) / expected_yield * 100

		anomaly_dict[region] = anomaly
		slope_dict[region] = slope

	anomaly_table = pd.DataFrame.from_dict(anomaly_dict, orient="index", columns=years)
	slopes = pd.Series(slope_dict, name="slope_t_ha_per_year")
	return anomaly_table, slopes

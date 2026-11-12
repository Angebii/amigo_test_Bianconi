# # Climate indices for the baseline
## Requires data_processed/yield_anomalies.csv with the columns yield_t_ha and area_ha:
## run notebook/01_explore_yields.py first.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from amigo_yield import compute_season_indices, read_era5, build_region_mask, compute_regional_means, add_harvest_year, compute_window_index

ERA5_DIR = Path("DATA/ERA5-Land")
BOUNDARIES_FILE = Path("DATA/OTHERS/france_regions_2016.gpkg")
PROCESSED_DIR = Path("data_processed")
DAILY_FILE = PROCESSED_DIR / "era5_daily_regional.csv"
DATASET_FILE = PROCESSED_DIR / "baseline_dataset.csv"

## Only the variables needed by the indices (all of them also exist in CMIP6)
INDEX_VARIABLES = ["tmax", "tmin", "precip"]

## Heat threshold from Noia Junior et al. (2025), who follow Nuttall et al. (2018)
HOT_DAY_THRESHOLD = 30   # degC


# %% Daily mean of every variable over each region
## The result is saved to a CSV file: if the file exists, it is read instead of recomputed.
if DAILY_FILE.exists():
	print("reading", DAILY_FILE)
	daily_climate = pd.read_csv(DAILY_FILE, parse_dates=["date"])
else:
	era5_variables, era5_summary = read_era5(ERA5_DIR, INDEX_VARIABLES)
	print(era5_summary.to_string(index=False))

	first_variable = era5_variables[INDEX_VARIABLES[0]]
	region_mask = build_region_mask(BOUNDARIES_FILE, first_variable["longitude"], first_variable["latitude"])

	daily_climate = compute_regional_means(era5_variables, region_mask)

	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
	daily_climate.to_csv(DAILY_FILE, index=False)
	print("saved", DAILY_FILE)

## Keep only the days where all the variables have data (precipitation covers a longer period)
has_all_variables = daily_climate[INDEX_VARIABLES].notna().all(axis=1)
daily_climate = daily_climate[has_all_variables]

print(daily_climate.head())
print("from", daily_climate["date"].min().date(), "to", daily_climate["date"].max().date())


# %% Harvest year of every day
## October to December belong to the harvest of the following year; August and September are removed
daily_climate = add_harvest_year(daily_climate)

## Daily mean temperature
daily_climate["tmean"] = (daily_climate["tmax"] + daily_climate["tmin"]) / 2

## Daily temperature range (tmax - tmin). Compared with its normal for the same months and region,
## a lower range suggests more cloud cover (less radiation), but also wetter soils or more humid air.
## Used only for May-June and standardised by region, so differences between seasons do not enter.
daily_climate["temperature_range"] = daily_climate["tmax"] - daily_climate["tmin"]


# %% How often is the heat threshold exceeded?
## The 30 degC threshold of the literature is defined on weather stations, where it is about the
## 90th percentile of tmax. Regional means are smoother, so the same threshold is exceeded less often.
heat_window = daily_climate[daily_climate["month"].isin([5, 6, 7])]
share_above = (heat_window["tmax"] > HOT_DAY_THRESHOLD).mean() * 100
percentile_90 = np.percentile(heat_window["tmax"], 90)

print("May-July, regional daily tmax")
print("  days above", HOT_DAY_THRESHOLD, "degC:", round(share_above, 2), "%")
print("  90th percentile:", round(percentile_90, 1), "degC")


# %% The indices
index_table = compute_season_indices(daily_climate)

## The interaction (index 2 x index 4) is computed in the baseline,
## after standardising the indices on the training years of each fold.

print(index_table.head())


# %% Check the indices: range, missing values and share of zeros
## An index that is almost always zero carries little information
summary_rows = []
index_names=index_table.drop(columns=["cod_regione", "harvest_year"]).columns
for name in index_names:
	values = index_table[name].dropna()
	row = {
		"index": name,
		"min": values.min(),
		"mean": values.mean(),
		"max": values.max(),
		"share_of_zeros_%": (values == 0).mean() * 100,
		"missing": index_table[name].isna().sum(),
	}
	summary_rows.append(row)

index_summary = pd.DataFrame(summary_rows)
print(index_summary.round(2).to_string(index=False))


# %% Join the indices with the yields
yields = pd.read_csv(PROCESSED_DIR / "yield_anomalies.csv")
yields = yields.rename(columns={"year": "harvest_year"})

baseline_dataset = yields.merge(index_table, on=["cod_regione", "harvest_year"], how="inner")

## Keep only the seasons where every index is available
has_all_indices = baseline_dataset[index_names].notna().all(axis=1)
baseline_dataset = baseline_dataset[has_all_indices]

print("harvest years:", baseline_dataset["harvest_year"].min(), "to", baseline_dataset["harvest_year"].max())
print("region-years:", len(baseline_dataset))


# %% Save the dataset
baseline_dataset.to_csv(DATASET_FILE, index=False)
print("saved", DATASET_FILE, "with columns:", list(baseline_dataset.columns))
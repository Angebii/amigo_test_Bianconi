# # Exploration of the Agreste yield workbook

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from amigo_yield import get_crop_table, compute_regional_anomalies

XLSX = Path("DATA/OTHERS/Agreste_SAA_cereali_mais_Francia_2000-2020 (1).xlsx")
PROCESSED_DIR = Path("data_processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

sheet_names = pd.ExcelFile(XLSX).sheet_names
print(sheet_names)


# %% Long-format regional sheet
long_table = pd.read_excel(XLSX, sheet_name="Regioni_long")

print("shape:", long_table.shape)
print("region types:", long_table["tipo"].unique())
print("indicators:", long_table["indicatore"].unique())
print("units:", long_table["unità"].unique())


# %% Wheat crops and their hierarchy levels
is_wheat = long_table["coltura"].str.contains("Blé")
wheat_codes = long_table.loc[is_wheat, ["codice", "livello", "coltura"]]
wheat_codes = wheat_codes.drop_duplicates()

print(wheat_codes.to_string(index=False))


# %% Keep metropolitan regions only
is_metropolitan = long_table["tipo"] == "Metropoli"
metro = long_table[is_metropolitan]

region_codes = metro.groupby("regione")["cod_regione"].first()
region_codes = region_codes.sort_values()
print(region_codes)


# %% Winter vs spring soft wheat: share of winter wheat in soft wheat area
soft_wheat_code = "1.11.1.11"
winter_soft_wheat_code = "1.11.1.11.1"
area_winter = get_crop_table(metro, winter_soft_wheat_code, "Superficie")
area_soft = get_crop_table(metro, soft_wheat_code, "Superficie")

## Share of winter wheat for every region and year
winter_share = area_winter / area_soft

mean_winter_share = winter_share.to_numpy().mean()
min_winter_share = winter_share.to_numpy().min()
print("mean winter share:", round(float(mean_winter_share), 3))
print("min winter share:", round(float(min_winter_share), 3))


# %% Missing values for soft wheat
is_soft_wheat = metro["codice"] == soft_wheat_code
soft_wheat_rows = metro[is_soft_wheat]

number_of_regions = 13
number_of_years = 21
number_of_indicators = 3
expected_rows = number_of_regions * number_of_years * number_of_indicators

print("rows:", len(soft_wheat_rows), "expected:", expected_rows)
print("missing values:", soft_wheat_rows["valore"].isna().sum())


# %% Mean soft wheat area per region (thousand ha)
mean_area_per_region = area_soft.mean(axis=1)
mean_area_per_region = mean_area_per_region / 1000
mean_area_per_region = mean_area_per_region.round(1)

print(mean_area_per_region.sort_values())

# %% Check that the yield column is production / area
yield_soft = get_crop_table(metro, soft_wheat_code, "Resa (calcolata)")
prod_soft = get_crop_table(metro, soft_wheat_code, "Produzione")
recomputed_yield = prod_soft / area_soft
difference = (recomputed_yield - yield_soft).abs()
max_difference = difference.to_numpy().max()

print("max |production/area - yield|:", max_difference)


# %% National yield (area-weighted) and linear trend
## National yield = total production / total area 
total_production = prod_soft.sum()
total_area = area_soft.sum()
national_yield = total_production / total_area

years = national_yield.index.to_numpy()
slope, intercept = np.polyfit(years, national_yield.to_numpy(), deg=1)
print(f"national trend slope: {slope:.4f} t/ha per year")

national_trend = slope * years + intercept
national_anomaly = ((national_yield - national_trend)/ national_trend) * 100

print("worst years (% from trend):")
print(national_anomaly.round(1).sort_values())

# %% Save the national anomaly
national_table = national_anomaly.reset_index()
national_table.columns = ["harvest_year", "anomaly_pct"]
national_table.to_csv(PROCESSED_DIR / "national_yield_anomaly.csv", index=False)
print(national_table.head())

# %% Regional anomalies (%) from a linear trend per region
## Trend fitted on all years: fine for exploration only (see amigo_yield.yields)
anomaly_table, regional_slopes = compute_regional_anomalies(yield_soft)

worst_years = [2001, 2003, 2007, 2011, 2016, 2020]
print(anomaly_table.round(0).to_string())
print("For the national wrost years")
print(anomaly_table[worst_years].round(0).to_string())


# %% Regional trend slopes (t/ha per year)
print(regional_slopes.round(3).sort_values())


# %% Heatmap of regional anomalies
## Rows = regions, largest wheat area on top. Columns = years.
## Red = yield loss, grey = on trend, blue = yield gain.
## Only anomalies of at least +/- label_threshold are written in the cells.
def plot_anomaly_heatmap(anomaly_table, region_order, title, label_threshold=5):
	data = anomaly_table.loc[region_order]
	values = data.to_numpy()

	## Symmetric colour scale centred on zero
	colour_map = LinearSegmentedColormap.from_list("loss_gain", ["#b2302f", "#f0efec", "#1c5cab"])
	limit = np.nanmax(np.abs(values))
	norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

	fig, ax = plt.subplots(figsize=(13, 6))
	image = ax.imshow(values, cmap=colour_map, norm=norm, aspect="auto")

	ax.set_xticks(range(len(data.columns)))
	ax.set_xticklabels(data.columns, rotation=90, fontsize=9)
	ax.set_yticks(range(len(data.index)))
	ax.set_yticklabels(data.index, fontsize=9)

	number_of_rows, number_of_columns = values.shape
	for row in range(number_of_rows):
		for column in range(number_of_columns):
			value = values[row, column]
			if abs(value) < label_threshold:
				continue

			## White text on dark cells, black text on light cells
			if abs(value) > 0.6 * limit:
				text_colour = "white"
			else:
				text_colour = "black"

			ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=7, color=text_colour)

	fig.colorbar(image, ax=ax, label="Yield anomaly (%)")
	ax.set_title(title)
	fig.tight_layout()
	plt.show()


region_order = area_soft.mean(axis=1).sort_values(ascending=False).index
plot_anomaly_heatmap(anomaly_table, region_order, "Soft wheat yield anomalies from the regional linear trend (%)")

# %% Save the anomalies in long format
## One row per region and year, with the region code used in the GeoPackage
anomaly_long = anomaly_table.stack()
anomaly_long = anomaly_long.reset_index()
anomaly_long.columns = ["regione", "year", "anomaly_pct"]

## region_codes: region name -> code, computed in the "Keep metropolitan regions only" cell
anomaly_long["cod_regione"] = anomaly_long["regione"].map(region_codes)
## Raw yield and soft wheat area of every region and year, for the baseline
yield_long = yield_soft.stack()
yield_long = yield_long.reset_index()
yield_long.columns = ["regione", "year", "yield_t_ha"]

area_long = area_soft.stack()
area_long = area_long.reset_index()
area_long.columns = ["regione", "year", "area_ha"]

anomaly_long = anomaly_long.merge(yield_long, on=["regione", "year"], how="left")
anomaly_long = anomaly_long.merge(area_long, on=["regione", "year"], how="left")
anomaly_long.to_csv(PROCESSED_DIR / "yield_anomalies.csv", index=False)
print(anomaly_long.head())
print("rows:", len(anomaly_long))

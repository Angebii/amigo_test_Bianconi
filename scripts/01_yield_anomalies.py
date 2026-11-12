## Yield anomalies of soft wheat from the Agreste workbook
## Writes:
##   data_processed/national_yield_anomaly.csv   (national anomaly, area-weighted)
##   data_processed/yield_anomalies.csv          (anomaly, yield and area of every region and year)
## The exploration of the workbook is in notebook/01_explore_yields.py.

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from amigo_yield import get_crop_table, compute_regional_anomalies

## Settings that change with new data, in config.yml at the root of the repository
with open("config.yml", encoding="utf-8") as config_file:
	pipeline_config = yaml.safe_load(config_file)

XLSX = Path(pipeline_config["paths"]["yields_file"])
PROCESSED_DIR = Path("data_processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SOFT_WHEAT_CODE = "1.11.1.11"


## Long-format regional sheet, metropolitan regions only
long_table = pd.read_excel(XLSX, sheet_name="Regioni_long")

is_metropolitan = long_table["tipo"] == "Metropoli"
metro = long_table[is_metropolitan]

## Region name -> region code used in the GeoPackage
region_codes = metro.groupby("regione")["cod_regione"].first()
region_codes = region_codes.sort_values()
print("regions:", len(region_codes))


## Soft wheat area, production and yield of every region and year
area_soft = get_crop_table(metro, SOFT_WHEAT_CODE, "Superficie")
prod_soft = get_crop_table(metro, SOFT_WHEAT_CODE, "Produzione")
yield_soft = get_crop_table(metro, SOFT_WHEAT_CODE, "Resa (calcolata)")


## National yield (area-weighted) and linear trend
## National yield = total production / total area
total_production = prod_soft.sum()
total_area = area_soft.sum()
national_yield = total_production / total_area

years = national_yield.index.to_numpy()
slope, intercept = np.polyfit(years, national_yield.to_numpy(), deg=1)
print(f"national trend slope: {slope:.4f} t/ha per year")

national_trend = slope * years + intercept
national_anomaly = ((national_yield - national_trend) / national_trend) * 100

print("worst years (% from trend):")
print(national_anomaly.round(1).sort_values().head())


## Save the national anomaly
national_table = national_anomaly.reset_index()
national_table.columns = ["harvest_year", "anomaly_pct"]
national_table.to_csv(PROCESSED_DIR / "national_yield_anomaly.csv", index=False)


## Regional anomalies (%) from a linear trend per region
## Trend fitted on all years. The baseline and the deep model fit the trend again
## on the training years of every fold (see amigo_yield.yields).
anomaly_table, regional_slopes = compute_regional_anomalies(yield_soft)

print("regional trend slopes (t/ha per year):")
print(regional_slopes.round(3).sort_values().to_string())


## Save the anomalies in long format
## One row per region and year, with the region code used in the GeoPackage
anomaly_long = anomaly_table.stack()
anomaly_long = anomaly_long.reset_index()
anomaly_long.columns = ["regione", "year", "anomaly_pct"]
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
print("rows:", len(anomaly_long))
print("saved", PROCESSED_DIR / "national_yield_anomaly.csv", "and", PROCESSED_DIR / "yield_anomalies.csv")

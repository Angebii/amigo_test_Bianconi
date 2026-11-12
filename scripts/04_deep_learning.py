## Deep learning model: 1D convolutional network on the daily climate of the season
## Requires:
##   data_processed/baseline_dataset.csv     (scripts/02_climate_indices.py)
##   data_processed/era5_daily_regional.csv  (scripts/02_climate_indices.py)
##   results/baseline/predictions.csv        (scripts/03_baseline.py), for the comparison
##
## Nested cross-validation grouped by year (see amigo_yield.deep):
##   outer loop - leave-one-year-out, the same test years as the Ridge baseline
##   inner loop - learning rate, weight decay and number of epochs chosen on the training years
##   final model - trained on all the training years, one model per seed

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from amigo_yield import build_daily_sequences, run_deep_leave_one_year_out, compute_metrics, YieldCNN, count_parameters

PROCESSED_DIR = Path("data_processed")
DATASET_FILE = PROCESSED_DIR / "baseline_dataset.csv"
DAILY_FILE = PROCESSED_DIR / "era5_daily_regional.csv"
BASELINE_PREDICTIONS_FILE = Path("results") / "baseline" / "predictions.csv"

RESULTS_DIR = Path("results") / "dl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

## Inputs: only variables that also exist in CMIP6, so the model can be used in Part C
VARIABLE_NAMES = ["tmax", "tmin", "precip"]

LOSS_THRESHOLD = -10   # yield anomaly in %

SETTINGS = {
	## Fixed architecture
	"fixed_parameters": {
		"number_of_filters": 2,
		"kernel_size": 7,      # one week of days
		"dropout": 0.2,
	},
	## Inner loop: ranges searched by Optuna (log scale)
	"learning_rate_range": [1e-4, 1e-2],
	"weight_decay_range": [1e-5, 1e-1],
	"number_of_trials": 25,
	"search_seed": 0,
	## Early stopping: maximum epochs, and epochs without improvement before stopping
	"max_epochs": 300,
	"patience": 20,
	## Final models: same settings, different random initial weights, batch order and dropout.
	## Predictions are averaged
	"seeds": [0, 1, 2, 3, 4],
	## Monte Carlo Dropout passes for the uncertainty
	"mc_samples": 0
}

## Same number of CPU threads on every machine, for comparable timings
torch.set_num_threads(1)

dataset = pd.read_csv(DATASET_FILE)
daily_climate = pd.read_csv(DAILY_FILE, parse_dates=["date"])
print("region-years:", len(dataset), "| harvest years:", dataset["harvest_year"].min(), "to", dataset["harvest_year"].max())


## Daily sequences: one block of 304 days x 3 variables for every region-year
sequences = build_daily_sequences(daily_climate, dataset, VARIABLE_NAMES)
print("sequences:", sequences.shape)

fixed = SETTINGS["fixed_parameters"]
example_model = YieldCNN(len(VARIABLE_NAMES), fixed["number_of_filters"], fixed["kernel_size"], fixed["dropout"])
number_of_parameters = count_parameters(example_model)
print("trainable parameters:", number_of_parameters)


## Nested cross-validation (this cell takes a while: one Optuna search per test year)
predictions, folds, learning_curves = run_deep_leave_one_year_out(dataset, sequences, SETTINGS)
print(folds.round(4).to_string(index=False))


## Learning curves: one summary over all the inner folds
## Every inner training (19 test years x 3 inner folds = 57 trainings) starts from a different level
## of error, so each curve is expressed as the change from its first epoch.
## Early stopping waits `patience` epochs, so every training lasts at least patience + 1 epochs:
## the summary uses these first epochs, where all the curves are available.
## Lines: median over the 57 trainings; bands: middle half (25th to 75th percentile).
## Training error going down while validation error stays flat or goes up = overfitting.
shown_epochs = SETTINGS["patience"] + 1
inner_curves = learning_curves[learning_curves["stage"].str.startswith("inner fold")]
inner_curves = inner_curves[inner_curves["epoch"] <= shown_epochs].copy()

## Change of every curve from its own first epoch
first_epoch = inner_curves.groupby(["test_year", "stage"])[["train_rmse", "validation_rmse"]].transform("first")
inner_curves["train_change"] = inner_curves["train_rmse"] - first_epoch["train_rmse"]
inner_curves["validation_change"] = inner_curves["validation_rmse"] - first_epoch["validation_rmse"]

## Median and middle half over the 57 trainings, epoch by epoch
summary = inner_curves.groupby("epoch")[["train_change", "validation_change"]].quantile([0.25, 0.5, 0.75]).unstack()

fig, ax = plt.subplots(figsize=(7, 4.5))
for column, colour, label in [("train_change", "#1c5cab", "training"), ("validation_change", "#e08a2c", "validation")]:
	ax.fill_between(summary.index, summary[(column, 0.25)], summary[(column, 0.75)], color=colour, alpha=0.2, lw=0)
	ax.plot(summary.index, summary[(column, 0.5)], color=colour, lw=2, label=label)
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(range(1, shown_epochs + 1, 2))
ax.set_xlabel("epoch")
ax.set_ylabel("change of RMSE from epoch 1 (points of anomaly)")
ax.set_title("Learning curves of the 57 inner trainings (median and middle half)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "learning_curves.png", dpi=150)

## Numbers for the report: median change after the last shown epoch
print(summary.loc[shown_epochs].round(2))


## One table with all the models, one row per region and year
comparison = pd.read_csv(BASELINE_PREDICTIONS_FILE)
comparison = comparison[["cod_regione", "regione", "harvest_year", "area_ha", "observed", "ridge"]]
comparison["zero"] = 0.0

cnn_columns = predictions[["cod_regione", "harvest_year", "observed", "cnn", "cnn_mc_std"]]
cnn_columns = cnn_columns.rename(columns={"observed": "observed_cnn"})
comparison = comparison.merge(cnn_columns, on=["cod_regione", "harvest_year"])

## The observed anomalies must be the same as in the baseline (same trend in every fold)
largest_difference = (comparison["observed"] - comparison["observed_cnn"]).abs().max()
print("largest difference of the observed anomalies from the baseline:", largest_difference)
comparison = comparison.drop(columns="observed_cnn")

MODEL_NAMES = ["zero", "ridge", "cnn"]

## Metrics of all the models
metric_rows = {}
for model_name in MODEL_NAMES:
	metric_rows[model_name] = compute_metrics(comparison["observed"], comparison[model_name], LOSS_THRESHOLD)

metrics = pd.DataFrame(metric_rows).T
print(metrics.round(3).to_string())


## Year by year: national anomaly (weighted by area) of every model
yearly_rows = []
for year, year_rows in comparison.groupby("harvest_year"):
	weights = year_rows["area_ha"]
	row = {
		"harvest_year": year,
		"observed": np.average(year_rows["observed"], weights=weights),
		"ridge": np.average(year_rows["ridge"], weights=weights),
		"cnn": np.average(year_rows["cnn"], weights=weights),
	}
	yearly_rows.append(row)

yearly = pd.DataFrame(yearly_rows)
print(yearly.round(1).to_string(index=False))

## In how many seasons is the CNN closer to the observations than the Ridge?
## (mean absolute error of the regions in every season)
error_ridge = (comparison["observed"] - comparison["ridge"]).abs().groupby(comparison["harvest_year"]).mean()
error_cnn = (comparison["observed"] - comparison["cnn"]).abs().groupby(comparison["harvest_year"]).mean()
seasons_better = (error_cnn < error_ridge).sum()
print("CNN better than Ridge in", seasons_better, "of", len(error_ridge), "seasons")

fig, ax = plt.subplots(figsize=(12, 4))
bar_width = 0.27
ax.bar(yearly["harvest_year"] - bar_width, yearly["observed"], width=bar_width, color="0.6", label="observed")
ax.bar(yearly["harvest_year"], yearly["ridge"], width=bar_width, color="#1c5cab", label="Ridge on indices")
ax.bar(yearly["harvest_year"] + bar_width, yearly["cnn"], width=bar_width, color="#e08a2c", label="1D CNN on daily climate")
ax.axhline(0, color="0.4", lw=0.8)
ax.axhline(LOSS_THRESHOLD, color="#b2302f", lw=0.8, ls="--")
ax.set_xticks(yearly["harvest_year"])
ax.set_xticklabels(yearly["harvest_year"], rotation=90)
ax.set_ylabel("yield anomaly (%)")
ax.set_title("National soft wheat yield anomaly (area-weighted): observed and predicted for unseen years")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "yearly.png", dpi=150)

## Save configuration and results of the deep model
config = {
	"model": "1D CNN on daily sequences with max pooling",
	"validation": "nested cross-validation grouped by year: outer leave-one-year-out, inner 3 groups of training years",
	"variables": VARIABLE_NAMES,
	"settings": SETTINGS,
	"trainable_parameters": number_of_parameters,
	"loss_threshold": LOSS_THRESHOLD,
	"total_minutes": float(folds["seconds"].sum() / 60),
	"peak_memory_mb": float(folds["memory_mb"].max()),
	"torch_threads": torch.get_num_threads(),
	"run_date": datetime.now().isoformat(timespec="seconds"),
}
with open(RESULTS_DIR / "config.json", "w") as config_file:
	json.dump(config, config_file, indent=2)

predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
folds.to_csv(RESULTS_DIR / "folds.csv", index=False)
learning_curves.to_csv(RESULTS_DIR / "learning_curves.csv", index=False)
comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
metrics.to_csv(RESULTS_DIR / "metrics.csv")
yearly.to_csv(RESULTS_DIR / "yearly.csv", index=False)


print("saved results in", RESULTS_DIR)

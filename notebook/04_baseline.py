# # Baseline: Ridge regression on climate indices
## Requires data_processed/baseline_dataset.csv: run notebook/03_climate_indices.py first.
##
## Validation: leave-one-year-out. For every test year, everything that depends on the
## data (yield trend, standardisation of the indices, choice of the Ridge penalty)
## is estimated on the training years only.

# %%
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from amigo_yield import prepare_fold, choose_alpha, compute_metrics

PROCESSED_DIR = Path("data_processed")
DATASET_FILE = PROCESSED_DIR / "baseline_dataset.csv"

## Configuration, tables and figures of the baseline, for the report and the comparison with the DL model
RESULTS_DIR = Path("results") / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAMES = [
	"precip_oct_jan",
	"tmean_nov_dec",
	"precip_mar_apr",
	"precip_may_jun",
	"temperature_range_may_jun",
	"tmax_may_jun",
	"hot_days_may_jul",
]
## Compound event of 2016: warm late autumn followed by a wet late spring
INTERACTION = ("tmean_nov_dec", "precip_may_jun")

## Candidate values of the Ridge penalty (larger = coefficients pulled more towards zero)
ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

LOSS_THRESHOLD = -10   # yield anomaly in %

dataset = pd.read_csv(DATASET_FILE)
print("region-years:", len(dataset), "| harvest years:", dataset["harvest_year"].min(), "to", dataset["harvest_year"].max())


# %% Leave-one-year-out: one Ridge model per test year
## Also a reference model that always predicts zero anomaly (a "normal" year)
all_years = sorted(dataset["harvest_year"].unique())
prediction_tables = []

for test_year in all_years:
	is_test = dataset["harvest_year"] == test_year
	train_rows = dataset[~is_test]
	test_rows = dataset[is_test]

	best_alpha = choose_alpha(train_rows, ALPHAS, INDEX_NAMES, INTERACTION)

	x_train, y_train, x_test, y_test = prepare_fold(train_rows, test_rows, INDEX_NAMES, INTERACTION)
	model = Ridge(alpha=best_alpha)
	model.fit(x_train, y_train)

	table = test_rows[["cod_regione", "regione", "harvest_year", "area_ha"]].copy()
	table["observed"] = y_test
	table["ridge"] = model.predict(x_test)
	table["zero"] = 0.0
	table["alpha"] = best_alpha
	prediction_tables.append(table)

	print(test_year, "alpha:", best_alpha)

predictions = pd.concat(prediction_tables, ignore_index=True)


# %% Metrics over all region-years
## rmse, skill against the zero model, correlation, loss hit rate and false alarm rate
## (definitions in amigo_yield.baseline.compute_metrics)
metric_rows = {}
for model_name in ["ridge", "zero"]:
	metric_rows[model_name] = compute_metrics(predictions["observed"], predictions[model_name], LOSS_THRESHOLD)

metrics = pd.DataFrame(metric_rows).T
print("seasons with a loss below", LOSS_THRESHOLD, "%:", (predictions["observed"] < LOSS_THRESHOLD).sum())
print(metrics.round(3).to_string())


# %% Year by year: national anomaly (weighted by soft wheat area), observed and predicted
yearly_rows = []
for year, year_rows in predictions.groupby("harvest_year"):
	weights = year_rows["area_ha"]
	row = {
		"harvest_year": year,
		"observed": np.average(year_rows["observed"], weights=weights),
		"ridge": np.average(year_rows["ridge"], weights=weights),
	}
	yearly_rows.append(row)

yearly = pd.DataFrame(yearly_rows)
print(yearly.round(1).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 4))
bar_width = 0.4
ax.bar(yearly["harvest_year"] - bar_width / 2, yearly["observed"], width=bar_width, color="0.6", label="observed")
ax.bar(yearly["harvest_year"] + bar_width / 2, yearly["ridge"], width=bar_width, color="#1c5cab", label="Ridge (leave-one-year-out)")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(yearly["harvest_year"])
ax.set_xticklabels(yearly["harvest_year"], rotation=90)
ax.set_ylabel("yield anomaly (%)")
ax.set_title("National soft wheat yield anomaly (area-weighted): observed and predicted for unseen years")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "yearly.png", dpi=150)
plt.show()


# %% Region-years: observed against predicted
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(predictions["observed"], predictions["ridge"], s=14, color="#1c5cab", alpha=0.7)

limit = np.abs(predictions[["observed", "ridge"]].to_numpy()).max() * 1.05
ax.plot([-limit, limit], [-limit, limit], color="0.4", lw=0.8)
ax.axvline(LOSS_THRESHOLD, color="#b2302f", lw=0.8, ls="--")
ax.axhline(LOSS_THRESHOLD, color="#b2302f", lw=0.8, ls="--")
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_xlabel("observed anomaly (%)")
ax.set_ylabel("predicted anomaly (%)")
ax.set_title("Region-years, Ridge (leave-one-year-out)")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "scatter.png", dpi=150)
plt.show()


# %% Coefficients of a model fitted on all years (for interpretation only)
## Each coefficient: change in yield anomaly (percentage points) for one standard deviation
## of the index above the regional normal. Not used for the evaluation above.
best_alpha_all = choose_alpha(dataset, ALPHAS, INDEX_NAMES, INTERACTION)
x_all, y_all, _, _ = prepare_fold(dataset, dataset, INDEX_NAMES, INTERACTION)
final_model = Ridge(alpha=best_alpha_all)
final_model.fit(x_all, y_all)

feature_names = INDEX_NAMES + [INTERACTION[0] + " x " + INTERACTION[1]]
coefficients = pd.Series(final_model.coef_, index=feature_names)
print("alpha:", best_alpha_all)
print(coefficients.round(2).sort_values().to_string())


# %% Save configuration and results of the baseline
config = {
	"model": "Ridge",
	"validation": "leave-one-year-out; alpha chosen with an inner leave-one-year-out on the training years",
	"index_names": INDEX_NAMES,
	"interaction": list(INTERACTION),
	"alphas": ALPHAS,
	"loss_threshold": LOSS_THRESHOLD,
	"harvest_years": [int(dataset["harvest_year"].min()), int(dataset["harvest_year"].max())],
	"region_years": int(len(dataset)),
	"alpha_all_years": float(best_alpha_all),
	"run_date": datetime.now().isoformat(timespec="seconds"),
}
with open(RESULTS_DIR / "config.json", "w") as config_file:
	json.dump(config, config_file, indent=2)

predictions.to_csv(RESULTS_DIR / "predictions.csv", index=False)
metrics.to_csv(RESULTS_DIR / "metrics.csv")
yearly.to_csv(RESULTS_DIR / "yearly.csv", index=False)
coefficients.to_csv(RESULTS_DIR / "coefficients.csv", header=["coefficient"])

print("saved results in", RESULTS_DIR)

# %%

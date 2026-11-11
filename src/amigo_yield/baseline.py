## Leave-one-year-out preparation, choice of the Ridge penalty and metrics.
## Everything that depends on the data (yield trend, standardisation of the indices)
## is estimated on the training rows only, and then applied to the test rows.

import numpy as np
from sklearn.linear_model import Ridge


## Linear yield trend of every region, fitted on the training rows only
def fit_trends(train_rows):
	trends = {}
	for region, region_rows in train_rows.groupby("cod_regione"):
		slope, intercept = np.polyfit(region_rows["harvest_year"], region_rows["yield_t_ha"], deg=1)
		trends[region] = (slope, intercept)
	return trends


## Relative anomaly (%) of the yield from the trend of its region
def compute_anomalies(rows, trends):
	anomalies = []
	for _, row in rows.iterrows():
		slope, intercept = trends[row["cod_regione"]]
		expected_yield = slope * row["harvest_year"] + intercept
		anomaly = (row["yield_t_ha"] - expected_yield) / expected_yield * 100
		anomalies.append(anomaly)
	return np.array(anomalies)


## Centre every index on the mean of its region and scale it with one standard deviation
## computed over all regions (both on the training rows). This is equivalent to region fixed
## effects: the model sees only the deviation from the regional normal, and one extra hot day
## counts the same in every region.
def standardise_indices(train_rows, test_rows, index_names):
	train_means = train_rows.groupby("cod_regione")[index_names].mean()

	## Deviations of the training rows from their regional means
	train_region_means = train_means.loc[train_rows["cod_regione"]].to_numpy()
	train_deviations = train_rows[index_names].to_numpy() - train_region_means

	## One standard deviation per index, over all regions
	overall_stds = train_deviations.std(axis=0, ddof=1)

	standardised = []
	for rows in [train_rows, test_rows]:
		region_means = train_means.loc[rows["cod_regione"]].to_numpy()
		values = (rows[index_names].to_numpy() - region_means) / overall_stds
		standardised.append(values)

	return standardised[0], standardised[1]


## Inputs of the Ridge regression: standardised indices and the interaction term.
## The standardisation uses the training rows only; the other rows can be any seasons with the
## same indices (held-out years, or seasons of a climate projection, which have no yield).
## interaction: pair of index names whose product is added as a last column (None for no interaction)
def prepare_inputs(train_rows, other_rows, index_names, interaction=None):
	x_train, x_other = standardise_indices(train_rows, other_rows, index_names)

	if interaction is not None:
		first = index_names.index(interaction[0])
		second = index_names.index(interaction[1])
		x_train = np.column_stack([x_train, x_train[:, first] * x_train[:, second]])
		x_other = np.column_stack([x_other, x_other[:, first] * x_other[:, second]])

	return x_train, x_other


## Inputs and target of one fold: anomalies, standardised indices and the interaction term.
## interaction: pair of index names whose product is added as a last column (None for no interaction)
def prepare_fold(train_rows, test_rows, index_names, interaction=None):
	trends = fit_trends(train_rows)
	y_train = compute_anomalies(train_rows, trends)
	y_test = compute_anomalies(test_rows, trends)

	x_train, x_test = prepare_inputs(train_rows, test_rows, index_names, interaction)

	return x_train, y_train, x_test, y_test


## Choose the Ridge penalty with an inner leave-one-year-out on the training years:
## for every candidate alpha, mean squared error of the predictions of every left-out year
def choose_alpha(train_rows, alphas, index_names, interaction=None):
	train_years = sorted(train_rows["harvest_year"].unique())
	mean_errors = {}

	for alpha in alphas:
		squared_errors = []
		for inner_test_year in train_years:
			is_inner_test = train_rows["harvest_year"] == inner_test_year
			inner_train = train_rows[~is_inner_test]
			inner_test = train_rows[is_inner_test]

			x_train, y_train, x_test, y_test = prepare_fold(inner_train, inner_test, index_names, interaction)
			model = Ridge(alpha=alpha)
			model.fit(x_train, y_train)
			predictions = model.predict(x_test)
			squared_errors.extend((predictions - y_test) ** 2)

		mean_errors[alpha] = np.mean(squared_errors)

	## Alpha with the lowest mean error
	best_alpha = None
	best_error = None
	for alpha, error in mean_errors.items():
		if best_error is None or error < best_error:
			best_alpha = alpha
			best_error = error

	return best_alpha


## Metrics of the predictions of one model.
## rmse: typical error in percentage points of anomaly
## skill_vs_zero: 1 - (RMSE model / RMSE zero model)^2; above 0 = better than predicting a normal year
## loss_hit_rate: share of loss seasons predicted as losses
## loss_false_alarm_rate: share of the other seasons wrongly predicted as losses
def compute_metrics(observed, predicted, loss_threshold):
	observed = np.asarray(observed)
	predicted = np.asarray(predicted)

	rmse = np.sqrt(np.mean((predicted - observed) ** 2))
	rmse_zero = np.sqrt(np.mean(observed ** 2))

	## The correlation is not defined for a model that always predicts the same value
	if np.std(predicted) > 0:
		correlation = np.corrcoef(observed, predicted)[0, 1]
	else:
		correlation = np.nan

	is_loss = observed < loss_threshold
	is_predicted_loss = predicted < loss_threshold
	hit_rate = (is_loss & is_predicted_loss).sum() / is_loss.sum()
	false_alarm_rate = (~is_loss & is_predicted_loss).sum() / (~is_loss).sum()

	return {
		"rmse": rmse,
		"skill_vs_zero": 1 - (rmse / rmse_zero) ** 2,
		"correlation": correlation,
		"loss_hit_rate": hit_rate,
		"loss_false_alarm_rate": false_alarm_rate,
	}

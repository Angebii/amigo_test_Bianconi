## Deep learning model: a small 1D convolutional network on the daily climate of the season.
## Every season (1 October to 31 July) is described by 304 days of tmax, tmin and precipitation.
##
## Validation: nested cross-validation grouped by year.
##   outer loop - leave-one-year-out: every year in turn is the test year, never used for any choice
##   inner loop - the training years are split into groups of years; each group in turn is the
##                validation set, used to choose learning rate, weight decay and number of epochs
##   final model - trained on all the training years with the chosen settings, one model per seed
## Everything that depends on the data (yield trend, scaling of inputs and target)
## is estimated on the training part of every split.

import copy
import time

import numpy as np
import optuna
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from amigo_yield.baseline import fit_trends, compute_anomalies
from amigo_yield.climate import add_harvest_year


## Days from 1 October to 31 July, without 29 February, so that every position
## is the same calendar day in every season
NUMBER_OF_DAYS = 304

## Blocks of the season after max pooling: 304 days / 10 blocks = about one month per block
NUMBER_OF_BLOCKS = 10

## Number of groups of training years in the inner loop
NUMBER_OF_INNER_FOLDS = 3


## Daily sequences of one season for every row of the dataset.
## daily: daily regional means with the columns cod_regione, date and the variables
## rows: one row per region and harvest year (columns cod_regione and harvest_year)
## Returns an array of shape (number of rows, 304 days, number of variables), in the order of rows.
def build_daily_sequences(daily, rows, variable_names):

	## Keep October to July and label every day with its harvest year
	season_days = add_harvest_year(daily)

	## Remove 29 February
	is_leap_day = (season_days["date"].dt.month == 2) & (season_days["date"].dt.day == 29)
	season_days = season_days[~is_leap_day]

	## Days in order within every region and season: from 1 October to 31 July
	season_days = season_days.sort_values(["cod_regione", "harvest_year", "date"])
	season_days = season_days.set_index(["cod_regione", "harvest_year"])

	## One block of 304 days for every row of the dataset
	sequences = np.zeros((len(rows), NUMBER_OF_DAYS, len(variable_names)), dtype=np.float32)
	for position, (region, year) in enumerate(zip(rows["cod_regione"], rows["harvest_year"])):
		season = season_days.loc[(region, year)]

		if len(season) != NUMBER_OF_DAYS:
			raise ValueError(f"region {region}, season {year}: {len(season)} days instead of {NUMBER_OF_DAYS}")

		sequences[position] = season[variable_names].to_numpy()

	return sequences


## Standardise the sequences with StandardScaler: for every variable, subtract the mean and
## divide by the standard deviation, both computed on the training sequences only
## (over all regions, seasons and days).
def standardise_sequences(train_sequences, test_sequences):

	number_of_variables = train_sequences.shape[2]

	## StandardScaler works on 2D tables: one row per day of every season, one column per variable
	train_table = train_sequences.reshape(-1, number_of_variables)
	test_table = test_sequences.reshape(-1, number_of_variables)

	## Mean and standard deviation from the training table only, then the same transformation for both
	scaler = StandardScaler()
	scaler.fit(train_table)
	train_table = scaler.transform(train_table)
	test_table = scaler.transform(test_table)

	## Back to (seasons, days, variables)
	train_standardised = train_table.reshape(train_sequences.shape)
	test_standardised = test_table.reshape(test_sequences.shape)

	return train_standardised.astype(np.float32), test_standardised.astype(np.float32)


## Inputs and target of one split (training part and held-out part).
## The sequences are in the same order as the rows.
def prepare_deep_fold(train_rows, test_rows, train_sequences, test_sequences):

	## Target: yield anomaly from the trend of the training years
	trends = fit_trends(train_rows)
	y_train = compute_anomalies(train_rows, trends)
	y_test = compute_anomalies(test_rows, trends)

	## Inputs: standardised daily sequences
	x_train, x_test = standardise_sequences(train_sequences, test_sequences)

	return x_train, y_train.astype(np.float32), x_test, y_test.astype(np.float32)


## Small 1D convolutional network.
## The filters slide along the 304 days and react to patterns of a few days (for example a hot week).
## Max pooling reduces the season to NUMBER_OF_BLOCKS blocks (about one month each), keeping the
## strongest response of each block. The flattened blocks keep the timing, so the last layer can weigh the same
## pattern differently in autumn and in June.
class YieldCNN(nn.Module):

	def __init__(self, number_of_variables, number_of_filters, kernel_size, dropout):
		super().__init__()
		self.convolution = nn.Conv1d(number_of_variables, number_of_filters, kernel_size, padding="same")
		self.activation = nn.ReLU()
		self.pooling = nn.AdaptiveMaxPool1d(NUMBER_OF_BLOCKS)
		self.dropout = nn.Dropout(dropout)
		self.output = nn.Linear(number_of_filters * NUMBER_OF_BLOCKS, 1)

	## x: (number of seasons, 304 days, number of variables)
	def forward(self, x):

		## Conv1d wants the variables as channels: (seasons, variables, days)
		x = x.transpose(1, 2)

		x = self.convolution(x)
		x = self.activation(x)
		x = self.pooling(x)
		x = x.flatten(start_dim=1)
		x = self.dropout(x)
		x = self.output(x)

		## One anomaly per season
		return x.squeeze(-1)


## Number of trainable parameters of a model
def count_parameters(model):
	total = 0
	for parameter in model.parameters():
		if parameter.requires_grad:
			total = total + parameter.numel()
	return total


## Root mean squared error, in points of anomaly
def compute_rmse(observed, predicted):
	return float(np.sqrt(np.mean((predicted - observed) ** 2)))


## Predicted anomalies (%) for the sequences x.
## The network predicts the standardised target; the target scaler of the model
## transforms the predictions back to points of anomaly.
## mc_samples = 0: dropout off, one deterministic prediction.
## mc_samples > 0: Monte Carlo Dropout, dropout kept on for mc_samples passes;
## returns the mean and the standard deviation of the passes.
def predict(model, x, mc_samples=0):
	inputs = torch.from_numpy(x)

	with torch.no_grad():
		if mc_samples == 0:
			model.eval()
			scaled_predictions = model(inputs).numpy()
			predictions = model.target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
			return predictions

		model.train()
		passes = []
		for _ in range(mc_samples):
			scaled_predictions = model(inputs).numpy()
			predictions = model.target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
			passes.append(predictions)
		model.eval()

	passes = np.array(passes)
	return passes.mean(axis=0), passes.std(axis=0)


## Train one network and record the training and validation loss after every epoch.
## parameters: number_of_filters, kernel_size, dropout, learning_rate, weight_decay
## With validation data: early stopping, i.e. training stops when the validation error has not
## improved for `patience` epochs, and the weights of the best epoch are kept.
## Without validation data: training runs for number_of_epochs epochs.
## The target is standardised with StandardScaler fitted on the training target (as the inputs);
## predict() transforms the predictions back to points of anomaly.
## Returns the model and the learning curve (one row per epoch).
def train_model(x_train, y_train, parameters, seed, number_of_epochs, patience, x_validation=None, y_validation=None, batch_size=32):

	## Same seed = same initial weights, same order of the batches, same dropout masks
	torch.manual_seed(seed)
	generator = torch.Generator().manual_seed(seed)

	model = YieldCNN(
		number_of_variables=x_train.shape[2],
		number_of_filters=parameters["number_of_filters"],
		kernel_size=parameters["kernel_size"],
		dropout=parameters["dropout"],
	)
	## Target standardised with its training mean and standard deviation
	model.target_scaler = StandardScaler()
	model.target_scaler.fit(y_train.reshape(-1, 1))
	scaled_targets = model.target_scaler.transform(y_train.reshape(-1, 1)).ravel()

	inputs = torch.from_numpy(x_train)
	targets = torch.from_numpy(scaled_targets).float()

	optimiser = torch.optim.Adam(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
	loss_function = nn.MSELoss()

	learning_curve = []
	best_error = None
	best_state = None
	epochs_without_improvement = 0

	for epoch in range(1, number_of_epochs + 1):

		## One pass over the training seasons, in random batches
		model.train()
		order = torch.randperm(len(inputs), generator=generator)
		for start in range(0, len(inputs), batch_size):
			batch = order[start:start + batch_size]
			optimiser.zero_grad()
			loss = loss_function(model(inputs[batch]), targets[batch])
			loss.backward()
			optimiser.step()

		## Training and validation error after this epoch
		train_error = compute_rmse(y_train, predict(model, x_train))
		validation_error = np.nan
		if x_validation is not None:
			validation_error = compute_rmse(y_validation, predict(model, x_validation))

		learning_curve.append({"epoch": epoch, "train_rmse": train_error, "validation_rmse": validation_error})

		## Early stopping on the validation error
		if x_validation is None:
			continue

		if best_error is None or validation_error < best_error:
			best_error = validation_error
			best_state = copy.deepcopy(model.state_dict())
			epochs_without_improvement = 0
		else:
			epochs_without_improvement = epochs_without_improvement + 1

		if epochs_without_improvement >= patience:
			break

	## Keep the weights of the best epoch
	if best_state is not None:
		model.load_state_dict(best_state)

	return model, pd.DataFrame(learning_curve)


## Inner loop: choose learning rate, weight decay and number of epochs on the training years only.
## The training years are split into NUMBER_OF_INNER_FOLDS groups; every Optuna trial is trained
## once per group, with that group as validation set and early stopping.
## The score of a trial is the mean of its best validation errors; the number of epochs is
## the mean of its best epochs. The training error at the best epochs is kept for comparison.
## Returns the chosen settings and the learning curves of the chosen trial (one per inner fold).
def search_hyperparameters(train_rows, train_sequences, settings):

	training_years = sorted(train_rows["harvest_year"].unique())
	year_groups = np.array_split(training_years, NUMBER_OF_INNER_FOLDS)

	## Learning curves of every trial, to keep those of the chosen one
	curves_of_trials = {}

	def objective(trial):
		parameters = dict(settings["fixed_parameters"])
		parameters["learning_rate"] = trial.suggest_float("learning_rate", settings["learning_rate_range"][0], settings["learning_rate_range"][1], log=True)
		parameters["weight_decay"] = trial.suggest_float("weight_decay", settings["weight_decay_range"][0], settings["weight_decay_range"][1], log=True)

		best_errors = []
		best_epochs = []
		train_errors = []
		trial_curves = []

		for inner_fold, validation_years in enumerate(year_groups, start=1):
			is_validation = train_rows["harvest_year"].isin(validation_years).to_numpy()

			x_train, y_train, x_validation, y_validation = prepare_deep_fold(
				train_rows[~is_validation],
				train_rows[is_validation],
				train_sequences[~is_validation],
				train_sequences[is_validation],
			)

			_, learning_curve = train_model(
				x_train, y_train, parameters, settings["search_seed"], settings["max_epochs"], settings["patience"],
				x_validation, y_validation,
			)

			best_row = learning_curve.loc[learning_curve["validation_rmse"].idxmin()]
			best_errors.append(best_row["validation_rmse"])
			best_epochs.append(best_row["epoch"])
			train_errors.append(best_row["train_rmse"])

			learning_curve.insert(0, "inner_fold", inner_fold)
			trial_curves.append(learning_curve)

		curves_of_trials[trial.number] = pd.concat(trial_curves, ignore_index=True)
		trial.set_user_attr("number_of_epochs", int(round(np.mean(best_epochs))))
		trial.set_user_attr("train_rmse", float(np.mean(train_errors)))

		return float(np.mean(best_errors))

	## Reproducible search: same seed, same sequence of trials
	optuna.logging.set_verbosity(optuna.logging.WARNING)
	sampler = optuna.samplers.TPESampler(seed=settings["search_seed"])
	study = optuna.create_study(direction="minimize", sampler=sampler)
	study.optimize(objective, n_trials=settings["number_of_trials"])

	chosen = dict(settings["fixed_parameters"])
	chosen.update(study.best_params)
	chosen["number_of_epochs"] = study.best_trial.user_attrs["number_of_epochs"]
	chosen["validation_rmse"] = study.best_value
	chosen["train_rmse"] = study.best_trial.user_attrs["train_rmse"]

	return chosen, curves_of_trials[study.best_trial.number]


## Outer loop: leave-one-year-out evaluation of the deep model.
## For every test year: inner search on the other years, then one final model per seed trained
## on all the other years for the chosen number of epochs, and predictions for the test year.
## Returns three tables:
##   predictions     - one row per region and year (observed, mean prediction, spread across seeds, MC Dropout spread)
##   folds           - one row per test year (chosen settings, training and validation RMSE, time, memory)
##   learning_curves - one row per test year, stage and epoch (inner folds of the chosen trial and final models)
def run_deep_leave_one_year_out(dataset, sequences, settings):

	process = psutil.Process()

	prediction_tables = []
	fold_rows = []
	curve_tables = []

	for test_year in sorted(dataset["harvest_year"].unique()):
		start_time = time.perf_counter()

		is_test = (dataset["harvest_year"] == test_year).to_numpy()
		train_rows = dataset[~is_test]
		test_rows = dataset[is_test]
		train_sequences = sequences[~is_test]
		test_sequences = sequences[is_test]

		## 1. Inner loop: settings chosen on the training years only
		chosen, inner_curves = search_hyperparameters(train_rows, train_sequences, settings)

		inner_curves.insert(0, "stage", "inner fold " + inner_curves["inner_fold"].astype(str))
		inner_curves = inner_curves.drop(columns="inner_fold")
		inner_curves.insert(0, "test_year", int(test_year))
		curve_tables.append(inner_curves)

		## 2. Final models on all the training years, one per seed
		x_train, y_train, x_test, y_test = prepare_deep_fold(train_rows, test_rows, train_sequences, test_sequences)

		seed_predictions = []
		mc_stds = []
		for seed in settings["seeds"]:
			model, final_curve = train_model(x_train, y_train, chosen, seed, chosen["number_of_epochs"], settings["patience"])
			seed_predictions.append(predict(model, x_test))
			# Monte Carlo Dropout spread, only if requested (mc_samples > 0)
			if settings["mc_samples"] > 0:
				_, mc_std = predict(model, x_test, mc_samples=settings["mc_samples"])
			else:
				mc_std = np.full(len(x_test), np.nan)
			mc_stds.append(mc_std)

			final_curve.insert(0, "stage", "final seed " + str(seed))
			final_curve.insert(0, "test_year", int(test_year))
			curve_tables.append(final_curve)

		seed_predictions = np.array(seed_predictions)

		## 3. Predictions of the test year: mean over the seeds and a measure of spread
		table = test_rows[["cod_regione", "regione", "harvest_year", "area_ha"]].copy()
		table["observed"] = y_test
		table["cnn"] = seed_predictions.mean(axis=0)
		table["cnn_mc_std"] = np.mean(mc_stds, axis=0)
		prediction_tables.append(table)

		## 4. Record of the fold
		fold_row = {
			"test_year": int(test_year),
			"learning_rate": chosen["learning_rate"],
			"weight_decay": chosen["weight_decay"],
			"number_of_epochs": chosen["number_of_epochs"],
			"train_rmse": chosen["train_rmse"],
			"validation_rmse": chosen["validation_rmse"],
			"seconds": time.perf_counter() - start_time,
			"memory_mb": process.memory_info().rss / 1024 ** 2,
		}
		fold_rows.append(fold_row)
		print(test_year, "| epochs:", chosen["number_of_epochs"], "| train RMSE:", round(chosen["train_rmse"], 2), "| validation RMSE:", round(chosen["validation_rmse"], 2), "| seconds:", round(fold_row["seconds"], 1))

	predictions = pd.concat(prediction_tables, ignore_index=True)
	folds = pd.DataFrame(fold_rows)
	learning_curves = pd.concat(curve_tables, ignore_index=True)

	return predictions, folds, learning_curves

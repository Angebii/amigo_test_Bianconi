## Tests of amigo_yield.deep: one test for each function

import numpy as np
import pandas as pd
import torch

from amigo_yield import build_daily_sequences, standardise_sequences, prepare_deep_fold, YieldCNN, count_parameters, predict, train_model


## Daily climate of one region for a season with 29 February (1 October 2003 to 31 July 2004).
## tmax is the day of the season: 0 on 1 October, 1 on 2 October, and so on.
def make_leap_season():
	dates = pd.date_range("2003-10-01", "2004-07-31", freq="D")
	daily = pd.DataFrame({
		"cod_regione": 11,
		"date": dates,
		"tmax": np.arange(len(dates), dtype=float),
	})
	return daily


## Two regions, five seasons each, yields growing by 0.1 t/ha per year (a perfect trend),
## and random daily sequences of 304 days x 2 variables, one per row of the dataset
def make_dataset_and_sequences():
	dataset = pd.DataFrame({
		"cod_regione": [11, 11, 11, 11, 11, 24, 24, 24, 24, 24],
		"harvest_year": [2001, 2002, 2003, 2004, 2005, 2001, 2002, 2003, 2004, 2005],
		"yield_t_ha": [7.0, 7.1, 7.2, 7.3, 7.4, 6.0, 6.1, 6.2, 6.3, 6.4],
	})

	## Random climate with a fixed seed, so every run gets the same values
	generator = np.random.default_rng(0)
	sequences = generator.normal(size=(10, 304, 2)).astype(np.float32)

	return dataset, sequences


## A leap season gives 304 days: 29 February is removed and the order of the days is kept
def test_build_daily_sequences():
	daily = make_leap_season()
	rows = pd.DataFrame({"cod_regione": [11], "harvest_year": [2004]})

	sequences = build_daily_sequences(daily, rows, ["tmax"])

	## 1 October is the first day, 31 July the last
	assert sequences.shape == (1, 304, 1)
	assert sequences[0, 0, 0] == 0.0
	assert sequences[0, -1, 0] == 304.0

	## 28 February (day 150) is followed by 1 March (day 152): day 151 was 29 February
	assert sequences[0, 150, 0] == 150.0
	assert sequences[0, 151, 0] == 152.0


## Changing the held-out year must not change the prepared training data (no leakage)
def test_prepare_deep_fold():
	dataset, sequences = make_dataset_and_sequences()

	is_test = (dataset["harvest_year"] == 2005).to_numpy()
	train_rows = dataset[~is_test]
	test_rows = dataset[is_test]

	## First preparation, with the original held-out year
	x_train_first, y_train_first, _, _ = prepare_deep_fold(train_rows, test_rows, sequences[~is_test], sequences[is_test])

	## Change the held-out year completely: half the yield, climate ten times larger
	changed_test_rows = test_rows.copy()
	changed_test_rows["yield_t_ha"] = changed_test_rows["yield_t_ha"] * 0.5
	changed_test_sequences = sequences[is_test] * 10

	## Second preparation, with the changed held-out year
	x_train_second, y_train_second, _, _ = prepare_deep_fold(train_rows, changed_test_rows, sequences[~is_test], changed_test_sequences)

	np.testing.assert_array_equal(x_train_first, x_train_second)
	np.testing.assert_array_equal(y_train_first, y_train_second)


## The network returns one value for every season
def test_yield_cnn():
	model = YieldCNN(number_of_variables=3, number_of_filters=2, kernel_size=7, dropout=0.2)
	x = torch.zeros((5, 304, 3))

	output = model(x)

	assert output.shape == (5,)


## Without validation data the model trains for all the epochs and records the training loss;
## with validation data early stopping ends the training at most `patience` epochs after the best epoch
def test_train_model():
	generator = np.random.default_rng(0)
	x_train = generator.normal(size=(20, 304, 3)).astype(np.float32)
	y_train = generator.normal(size=20).astype(np.float32)
	x_validation = generator.normal(size=(10, 304, 3)).astype(np.float32)
	y_validation = generator.normal(size=10).astype(np.float32)
	parameters = {"number_of_filters": 2, "kernel_size": 7, "dropout": 0.2, "learning_rate": 0.01, "weight_decay": 0.0}

	_, curve_without_validation = train_model(x_train, y_train, parameters, 0, 15, 5)
	assert len(curve_without_validation) == 15
	assert curve_without_validation["validation_rmse"].isna().all()

	_, curve_with_validation = train_model(x_train, y_train, parameters, 0, 200, 5, x_validation, y_validation)
	best_epoch = curve_with_validation.loc[curve_with_validation["validation_rmse"].idxmin(), "epoch"]
	assert len(curve_with_validation) <= best_epoch + 5




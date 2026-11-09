## Tests of amigo_yield.baseline: anomalies, standardisation, absence of leakage, metrics

import numpy as np
import pandas as pd
import pytest

from amigo_yield import prepare_fold


## Small hand-made dataset used by the tests
## Two regions observed for five years (2001 to 2005):
## - yield grows by exactly 0.1 t/ha per year, so it lies on a perfect linear trend
## - rain is a climate index that changes every year
def make_dataset():
	rows = [
		[11, 2001, 7.0,  80.0],
		[11, 2002, 7.1,  90.0],
		[11, 2003, 7.2, 100.0],
		[11, 2004, 7.3, 110.0],
		[11, 2005, 7.4, 120.0],
		[24, 2001, 6.0, 280.0],
		[24, 2002, 6.1, 290.0],
		[24, 2003, 6.2, 300.0],
		[24, 2004, 6.3, 310.0],
		[24, 2005, 6.4, 320.0],
	]
	columns = ["cod_regione", "harvest_year", "yield_t_ha", "rain"]
	dataset = pd.DataFrame(rows, columns=columns)
	return dataset


## Changing the test year must not change the prepared training data
## If trend or standardisation used the test year, the training data would change
def test_prepare_fold():

	dataset = make_dataset()
	index_names = ["rain"]

	## Training years 2001 to 2004, test year 2005
	train_rows = dataset[dataset["harvest_year"] < 2005]
	test_rows = dataset[dataset["harvest_year"] == 2005].copy()

	## First preparation, with the original test year
	train_first, _, _, _ = prepare_fold(train_rows, test_rows, index_names)

	## Change the test year completely: half the yield, ten times the rain
	changed_test_rows = test_rows.copy()
	changed_test_rows["yield_t_ha"] = changed_test_rows["yield_t_ha"] * 0.5
	changed_test_rows["rain"] = changed_test_rows["rain"] * 10

	## Second preparation, with the changed test year
	train_second, _, _, _ = prepare_fold(train_rows, changed_test_rows, index_names)

	## The prepared training data must be identical
	np.testing.assert_array_equal(train_first, train_second)



## Shared functions for the analysis of soft wheat yields and climate in France.
## The functions are organised in modules by topic:
##   yields    - Agreste yield tables and regional yield anomalies
##   climate   - ERA5-Land reading, regional means, harvest year
##   indices   - climate indices over windows of months
##   baseline  - baseline model
##   deep      - deep model
##   cmip6     - CMIP6 reading, interpolation on the ERA5-Land grid, bias correction


from amigo_yield.yields import (
	get_crop_table,
	compute_regional_anomalies,
)
from amigo_yield.climate import (
	read_era5,
	convert_units,
	get_short_name,
	compute_wind_speed,
	build_region_mask,
	compute_regional_means,
	add_harvest_year,
)
from amigo_yield.indices import (
	compute_window_index,
	compute_season_indices,
	INDEX_NAMES,
	HOT_DAY_THRESHOLD,
)
from amigo_yield.baseline import (
	fit_trends,
	compute_anomalies,
	standardise_indices,
	prepare_inputs,
	prepare_fold,
	choose_alpha,
	compute_metrics,
)

from amigo_yield.deep import (
	build_daily_sequences,
	standardise_sequences,
	prepare_deep_fold,
	YieldCNN,
	count_parameters,
	compute_rmse,
	predict,
	train_model,
	search_hyperparameters,
	run_deep_leave_one_year_out,
)

from amigo_yield.cmip6 import (
	read_cmip6_variable,
	interpolate_to_grid,
	train_bias_correction,
	apply_bias_correction,
	CMIP6_NAMES,
)

## Shared functions for the analysis of soft wheat yields and climate in France.
## The functions are organised in modules by topic:
##   yields    - Agreste yield tables and regional yield anomalies
##   climate   - ERA5-Land reading, regional means, harvest year
##   indices   - climate indices over windows of months


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
)
from amigo_yield.baseline import (
	fit_trends,
	compute_anomalies,
	standardise_indices,
	prepare_fold,
	choose_alpha,
	compute_metrics,
)

## Single entry point of the pipeline: runs every script in order and regenerates
## data_processed/ and results/ from the input data in DATA/.
## Usage, from the root of the repository:
##   python run_pipeline.py
## The notebooks in notebook/ are exploratory and are not part of the pipeline.

import os
import subprocess
import sys
import time

## Order of the steps: every step reads the outputs of the previous ones
STEPS = [
	"scripts/01_yield_anomalies.py",
	"scripts/02_climate_indices.py",
	"scripts/03_baseline.py",
	"scripts/04_deep_learning.py",
	"scripts/05_cmip6_historical.py",
	"scripts/06_cmip6_future.py",
]

## Figures are saved to file and not shown on screen
environment = os.environ.copy()
environment["MPLBACKEND"] = "Agg"

pipeline_start = time.time()

for step in STEPS:
	print("Running", step, flush=True)
	start_time = time.time()

	## Every step runs in its own process: memory is freed at the end of the step
	## and a failing step stops the whole pipeline
	subprocess.run([sys.executable, step], env=environment, check=True)

	elapsed_minutes = (time.time() - start_time) / 60
	print("Finished", step, "in", round(elapsed_minutes, 1), "minutes", flush=True)

total_minutes = (time.time() - pipeline_start) / 60
print("Pipeline finished in", round(total_minutes, 1), "minutes")

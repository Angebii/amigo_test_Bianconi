# amigo_test_Bianconi

Climate drivers of soft wheat yield losses in 13 French regions (2001–2019) and their possible evolution under the CMIP6 SSP5-8.5 scenario.

The pipeline computes regional yield anomalies, builds climate indices from ERA5-Land, compares a Ridge regression baseline with a 1D convolutional neural network, corrects three CMIP6 models with Quantile Delta Mapping and applies the baseline to two future windows (2031–2050 and 2071–2090).

## Repository structure

```
DATA/                 input data (ERA5-Land, CMIP6, regional boundaries, Agreste yields)
src/amigo_yield/      package with the functions used by the scripts and the notebooks
scripts/              steps of the pipeline, numbered in execution order
notebook/             exploratory analysis, not needed to reproduce the results
tests/                tests of the core functions
data_processed/       intermediate files written by the pipeline
results/              tables and figures used in the report
config.yml            input paths, present period, CMIP6 models, scenario and future windows
environment.yml       conda environment
pyproject.toml        installation of the package amigo_yield
run_pipeline.py       single entry point
AI_USAGE.md           use of AI tools during the work
```

## Environment

```
conda env create -f environment.yml
conda activate amigo_test
pip install -e .
```

The last command installs the package `amigo_yield` from `src/`, so the scripts, the notebooks and the tests can import it.

## Running the pipeline

From the root of the repository:

```
python run_pipeline.py
```

The script runs the steps in this order, each one in its own Python process, and stops at the first error. Figures are saved to file and not shown on screen. To keep a record of the run:

```
python run_pipeline.py > pipeline.log 2>&1
```

| Step | Output | Time |
|---|---|---|
| 01_yield_anomalies | regional and national yield anomalies (`data_processed/`) | 0.2 min |
| 02_climate_indices | daily regional climate and seasonal indices (`data_processed/`) | 0.6 min |
| 03_baseline | Ridge baseline, nested cross-validation (`results/baseline/`) | 1.0 min |
| 04_deep_learning | 1D CNN, nested cross-validation (`results/dl/`) | 8.8 min |
| 05_cmip6_historical | bias correction of the historical CMIP6 runs and comparison with ERA5-Land (`results/cmip6/`) | 8.3 min |
| 06_cmip6_future | bias correction of SSP5-8.5, future indices and yield anomalies (`results/cmip6/`) | 15.8 min |
| Total | | 34.7 min |

The notebooks in `notebook/` read the files written by the pipeline and can be opened after a run.

## Tests

```
pytest
```

## Running on new data

The settings that change with new data are in `config.yml`. The methodological choices (indices, loss threshold, model settings, bias correction) are set in the scripts and described in the report.

- New climate scenario: place the CMIP6 files in `DATA/CMIP6/<model>/<experiment>/` and change `experiment` and `label` in `config.yml`. The historical runs used for the calibration do not change.
- Other future windows: change `windows` in `config.yml`.
- Other climate models: change `models` in `config.yml`. The colours of the models in the figures of steps 05 and 06 are set in the scripts for the three current models.
- Additional year of yields: replace the Agreste file in `DATA/OTHERS`, update `yields_file` and the `present` period in `config.yml`. The ERA5-Land files must cover the new growing season (October to July).
- Steps 05_cmip6_historical and 06_cmip6_future save the regional CMIP6 series in `data_processed/` and read them back on the next run. The future series carry the name of the scenario, so a new scenario is always computed. Delete `data_processed/` before running the pipeline on new data or after changing the code, so every file is computed again.

## Hardware and computational cost

| | |
|---|---|
| CPU | Intel Core i7-8750H, 2.20 GHz, 6 cores |
| RAM | 16 GB |
| GPU | not used |
| Operating system | Windows 11 Home |
| PyTorch threads | 1 |

| Model | Trainable parameters | Training time | Peak memory |
|---|---|---|---|
| 1D CNN | 65 | 8.6 min | 551 MB |
| Ridge baseline | 9 (8 coefficients and intercept) | 48 s | 444 MB |

Training time covers the whole nested cross-validation of each model. For the CNN, this means 19 outer folds (one for each year), each with a hyperparameter search of 25 Optuna trials and a final training with 5 seeds. For the Ridge, it includes the choice of the penalty with an inner validation in each fold. Peak memory is the resident memory of the Python process: for the CNN the maximum of the values recorded at the end of each outer fold, for the Ridge the value at the end of the cross-validation. The values are saved in `results/dl/config.json` and `results/baseline/config.json` at every run.

## Use of AI tools

See `AI_USAGE.md`.

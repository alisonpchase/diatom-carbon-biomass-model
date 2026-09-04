# Diatom Carbon Biomass Model

This repository contains the codes and files needed for the training, validating, and running inference with the diatom carbon biomass model developed by Chase et al. (preprint).

## Contents

- `model_config.py` — model feature names, pigment ratio definitions, model target, and RF hyperparameters
- `prepare_features.py` — feature preparation and log-transform pipeline
- `train_evaluate_model.py` — model evaluation and diagnostic workflow
- `train_model_final_export.py` — final RF training and ONNX export
- `run_model_inference.py` — run inference to calculate diatom carbon biomass on input features
- `trained_models/` — exported ONNX model and companion uncertainty/domain bundle

## Requirements

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate diatom-model-release
```

## Model workflow

### 1) Train and validate the model

```bash
python train_evaluate_model.py \
  --training-data /path/to/training_data.csv \
  --figures-dir /path/to/figures
```

Required input file:
- `--training-data` must point to the training CSV before the script will run.

Optional environment-variable overrides are still supported for compatibility, but explicit CLI (command-line interface) arguments are preferred.

```bash
DIATOM_TRAINING_DATA=/path/to/data.csv DIATOM_FIGURES_DIR=/path/to/figures \
python train_evaluate_model.py --training-data /path/to/data.csv --figures-dir /path/to/figures
```

### 2) Train the final model and export in ONNX format

```bash
python train_model_final_export.py \
  --training-data /path/to/training_data.csv \
  --output-model /path/to/trained_models \
  --model-name my_diatom_model.onnx
```

Required input files:
- `--training-data` must point to the CSV used to fit the final model.
- `--output-model` is required so the model and companion bundle are written to the desired location. It may be an ONNX file path or an existing/output directory.
- `--model-name` names the ONNX file when `--output-model` is a directory; the `.onnx` suffix is added if needed.

### 3) Run model inference

```bash
python run_model_inference.py input.csv output.csv /path/to/trained_model.onnx
```

The model path is required. It can also be supplied through `DIATOM_INFERENCE_MODEL`:

```bash
DIATOM_INFERENCE_MODEL=/path/to/trained_model.onnx \
python run_model_inference.py input.csv
```

### Example usage

```bash
python train_evaluate_model.py \
  --training-data /path/to/ifcb_enviro_09Mar26.csv \
  --figures-dir ./figures

python train_model_final_export.py \
  --training-data /path/to/ifcb_enviro_09Mar26.csv \
  --output-model ./trained_models \
  --model-name diatom_model.onnx

python run_model_inference.py /path/to/input.csv /path/to/output.csv \
  ./trained_models/diatom_model.onnx
```

## Input format

The model expects a CSV with at least these columns:

- `chla`
- `chlb`
- `chlc`
- `ppc`

The script will automatically derive:

- `chlb_a = chlb / chla`
- `chlc_a = chlc / chla`
- `ppc_a = ppc / chla`

and then apply the same preprocessing used during training.

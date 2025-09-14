Modeling Irregularly-sampled and Multivariate Time Series
(IMTS) is crucial across a variety of applications where different sets of variates may be missing at different time-steps due
to sensor malfunctions or high data acquisition costs. Existing
approaches for IMTS either consider a two-stage impute-thenmodel framework or involve specialized architectures specific
to a particular model and task. We perform a series of experiments to derive novel insights about the performance of IMTS
methods on a variety of semi-synthetic and real-world datasets
for both classification and forecasting. We also introduce
Missing Feature-aware Time Series Modeling (MissTSM)
or MissTSM, a novel model-agnostic and imputation-free approach for IMTS modeling. We show that MissTSM shows
competitive performance compared to other IMTS approaches,
especially when the amount of missing values is large and the
data lacks simplistic periodic structures—conditions common
to real-world IMTS applications.

Paper - [Investigating a Model-Agnostic and Imputation-Free Approach for Irregularly-Sampled Multivariate Time-Series Modeling](https://arxiv.org/abs/2502.15785)


## Repository Structure

```plaintext
project_root/
├── forecasting/
│   ├── misstsm_mae/              # MissTSM integrated with MAE model
│   ├── misstsm_itransformer/     # MissTSM integrated with iTransformer model
│   ├── misstsm_patchtst/         # MissTSM integrated with PatchTST model
│   └── baselines/                # other Baseline models
|   └── data/                     # contain original forecasting and synthetically generated data
│
└── classification/
    ├── IMTS/                     # MissTSM integrated with MAE model applied to IMTS classification datasets
    └── synthetic_masked/         # MissTSM integrated with MAE model applied to synthetically masked classification datasets
    └── data/                     # contain original and synthetically generated data
```

- Each main directory contains a `scripts/` folder with relevant shell scripts for running experiments.


## Running Forecasting Experiments

### 1. MissTSM-MAE

Navigate to `misstsm_mae/` and run the desired shell script:

```sh
bash ./scripts/ETTh2_masking.sh "p6" 0 0 mcar "96,192"
```
where
- "p6" refers to the root_paths
- 0 refers to the gpu device
- 0 refers to the trial
- mcar refers to the masking type
- "96,192" refers to the prediction lengths

**Parameters:**
- --root_paths: Specifies the masked data fraction.  
  - p6: 60% MCAR masked data  
  - a6: 60% periodic masked data  
  - Prefix p = MCAR, Prefix a = periodic
- --trial: Indicates which version of the masked file to use.
- --maskingtype: Type of masking (mcar or periodic).
- --pred_len_list: List of prediction lengths (may be hardcoded in some scripts).

### 2. MissTSM-iTransformer & MissTSM-PatchTST

Navigate to the respective `misstsm_itransformer/` or `misstsm_patchtst` directory. There are two types of scripts:

a. *_masked.sh (MissTSM Integrated)

Example:
```sh
bash ./scripts/iTransformer_ETTh2_masked.sh \
  --root_paths "p7" \
  --devices 0 \
  --trial 0 \
  --maskingtype mcar\
  --run_name experiment1 \
  --mtsm_norm 1 \
  --embed tfi \
  --layernorm 1 \
  --inverted 1 \
  --skip 1 \
  --use_misstsm 1
```

**Arguments:**
- run_name: Name of the run/task.
- mtsm_norm: Use RevIN normalization (set to 1).
- embed: Type of embedding (set to tfi ).
- layernorm: Activate layer normalization (set to 1).
- inverted: (iTransformer only) Use inverted MissTSM (set to 1).
- skip: Add skip connection (1 to enable).
- use_misstsm: Set to 1 to use MissTSM; set to 0 to revert to original model.

b. Imputation Scripts

Run the original MTS model on imputed files:

```sh
bash ./scripts/iTransformer_ETTh2_imputation.sh --root_paths p6 --devices 0 --trial 0 --maskingtype mcar --imputation spline
```

**Arguments:**
- root_paths: Masked data fraction (as above).
- maskingtype: Type of masking.
- imputation: Imputation method (spline or saits).

### 3. Baselines

Baseline models and scripts are present in the baselines/ directory. Refer to individual scripts for specific instructions and parameters.

---

## Running Classification Experiments

### 1. MissTSM-MAE on IMTS Datasets

Navigate to `classification/IMTS/` and run the desired shell script:

```sh
python train.py --dataset P12 --epochs 100
```
The processed data is obtained from https://physionet.org/content/challenge-2012/1.0.0/, and https://physionet.org/content/challenge-2019/1.0.0/, respectively.

### 2. MissTSM-MAE on Synthetic Masked Datasets

Navigate to `classification/synthetic_masked/` and run:

```sh
bash ./scripts/run_all.sh
```
Comment out the dataset that you do not wish to run


---

## Requirements

- Python (>=3.9)
- See `requirements.txt` for Python dependencies.

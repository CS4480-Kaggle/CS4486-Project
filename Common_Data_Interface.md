# Iceberg Classification Common Data Interface

This document defines the shared data contract used by Model A, Model B,
Model C, and the final ensemble. It separates raw competition data, model-ready
inputs, and prediction outputs so that each model can be developed independently
without changing the meaning of shared fields.

## Raw data schema

Training records contain:


| Field        | Type                 | Meaning                               |
| -------------- | ---------------------- | --------------------------------------- |
| `id`         | string               | Unique sample identifier              |
| `band_1`     | list of 5625 numbers | First 75 x 75 SAR band                |
| `band_2`     | list of 5625 numbers | Second 75 x 75 SAR band               |
| `inc_angle`  | numeric or missing   | Radar incidence angle                 |
| `is_iceberg` | integer              | Training label: ship `0`, iceberg `1` |

Test records contain the same fields except `is_iceberg`. 

## Canonical sample interface

Every sample exposes these fields before model-specific feature engineering:

```text
id: string
image_2ch: float32 array with shape [2, 75, 75]
inc_angle_raw: float32 or missing
angle_missing: boolean
target: float32 or absent for test data
fold: integer 0 to 4 for training data
```

The image channel order is fixed:

```text
image_2ch[0] = band_1
image_2ch[1] = band_2
```

`angle_missing` records whether the original value was missing. 

## PyTorch interface

The supported image configurations are:


| Name             | Channels | Definition                                  |
| ------------------ | ---------: | --------------------------------------------- |
| `raw_2ch`        |        2 | `band_1`, `band_2`                          |
| `mean_3ch`       |        3 | `band_1`, `band_2`, `(band_1 + band_2) / 2` |
| `difference_3ch` |        3 | `band_1`, `band_2`, `band_1 - band_2`       |

The final deep-learning branches use `mean_3ch`.

A dataset item follows this structure:

```python
{
    "id": str,
    "image": torch.float32,         # [C, 75, 75]
    "angle": torch.float32,         # [1]
    "angle_missing": torch.float32, # [1]
    "target": torch.float32,        # [1], train only
}
```

A batch has the corresponding leading batch dimension:

```text
image: [batch_size, C, 75, 75]
angle: [batch_size, 1]
angle_missing: [batch_size, 1]
target: [batch_size, 1]
```

Image tensors use `float32`. Models return one logit per sample with shape
`[batch_size, 1]` before the sigmoid operation.

## LightGBM feature interface

Model C transforms canonical images into tabular features. A feature table
preserves one row per sample and keeps `id` for alignment, but `id` is not used
as a predictive feature.

Training features include:

```text
id
fold
target
feature columns...
```

Test features include:

```text
id
feature columns...
```

Train and test feature columns have the same names and order. PCA and all
learned preprocessing are fitted inside each training fold only.

## Shared five-fold interface

The official split is stored at `artifacts/folds.csv` and contains `id,fold`.

Rules:

- use stratified five-fold cross-validation;
- fold values are exactly `0, 1, 2, 3, 4`;
- every training ID appears exactly once;
- every model reuses the same `id -> fold` mapping;
- no model branch regenerates its own official folds.

## Fold-safe preprocessing

For each fold, preprocessing is fitted with the training portion only. The
saved training-fold statistics are applied to the validation portion and test
set.

Image normalization stores one mean and standard deviation per channel. Angle
preprocessing stores `angle_median`, `angle_mean`, and `angle_std`.

Angle processing order:

1. create `angle_missing` from the original value;
2. fill missing values with the current training-fold median;
3. normalize with the current training-fold mean and standard deviation.

Validation and test observations never influence fitted statistics.

## Augmentation interface

Random augmentation is used only for training samples. Validation and test
inputs remain deterministic. Spatial transforms are applied identically to all
channels of the same sample. Angle values, missing indicators, labels, IDs, and
fold assignments are not changed by image augmentation.

Test-time augmentation averages probabilities from deterministic views and is
recorded as part of the run configuration.

## OOF prediction interface

Every final model writes one out-of-fold prediction per training sample:

```text
id,fold,target,prediction,model_name,run_id
```

Requirements:

- exactly 1604 rows and 1604 unique IDs;
- each prediction is produced by a model that did not train on that row;
- probabilities are finite and within `[0, 1]`;
- `fold` and `target` agree across model branches.

## Test prediction interface

Every final model writes:

```text
id,prediction,model_name,run_id
```

Requirements:

- exactly 8424 rows and 8424 unique IDs;
- probabilities are finite and within `[0, 1]`;
- alignment uses `id`, never assumed row order alone.

Internal prediction files always use the column name `prediction`.

## Kaggle submission interface

Submission files use exactly:

```text
id,is_iceberg
```

`is_iceberg` is copied from the internal `prediction` column only at the final
submission step. A valid submission has 8424 unique test IDs and probabilities
within `[0, 1]`.

## Ensemble interface

Ensemble inputs are aligned by `id`. Static blending learns non-negative
weights that sum to one. Logistic stacking uses model logits as meta-features.
Both learned ensemble methods use cross-fitting so that each OOF ensemble
prediction is generated by a meta-model that was not fitted on that held-out
fold.

## Required validation checks

Before accepting model output, verify:

- expected columns and row counts;
- unique IDs;
- probability range and finite values;
- exact shared fold mapping;
- identical targets across OOF files;
- identical test ID sets across branches;
- submission columns and row count;
- reproducible OOF log loss from saved predictions.

Automated checks are stored in `tests/test_prediction_interface.py` and
`tests/test_ensemble_artifacts.py`.

## Reproducibility fields

Every formal run records:

```text
run_id
model name and architecture
input configuration
metadata configuration
fold file
base seed and fold seeds
epochs and batch size
optimizer and learning rate
weight decay and dropout
augmentation and test-time augmentation
pretrained checkpoint
best epoch per fold
fold log loss
overall OOF log loss
OOF and test output paths
Git commit
```

Earlier formal runs are not overwritten. Every distinct configuration uses a
unique `run_id`.

# Competition data

Raw competition files are intentionally excluded from Git.

Accept the Kaggle competition rules before downloading the files. Place the
downloaded and extracted files in the `raw/` directory.

Expected files:

```text
data/raw/train.json
data/raw/test.json
data/raw/sample_submission.csv
```

The training data must contain `id`, `band_1`, `band_2`, `inc_angle`, and
`is_iceberg`. The test data contains the same fields except `is_iceberg`.

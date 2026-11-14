# Competition data / 竞赛数据

Raw competition files are intentionally excluded from Git.

原始竞赛数据不会上传到仓库。请先在 Kaggle 接受比赛规则，再将下载并解压后的文件放在本目录下的 `raw/` 文件夹中。

Expected files / 预期文件：

```text
data/raw/train.json
data/raw/test.json
data/raw/sample_submission.csv
```

The training data must contain `id`, `band_1`, `band_2`, `inc_angle`, and `is_iceberg`. The test data contains the same fields except `is_iceberg`.

# CS4486 Mini Project: Statoil/C-CORE Iceberg Classification

This repository integrates the three model tracks developed for the CS4486 project:

- Model A: optimized small CNN;
- Model B: ImageNet-pretrained ResNet18 for 75 x 75 SAR images;
- Model C: multi-seed LightGBM with fold-safe PCA features;
- Final ensemble: equal averaging, cross-fitted static blending, and cross-fitted logistic stacking.

Raw Kaggle data is not committed to Git. All formal experiments reuse the same
stratified five-fold split and are evaluated with out-of-fold log loss.

## Final results

Lower log loss is better.


| Candidate                            | OOF log loss | Kaggle public | Kaggle private |
| -------------------------------------- | -------------: | --------------: | ---------------: |
| Model A: A2-14 Wider SimpleCNN       |     0.267826 |       0.21774 |        0.21644 |
| Model B: E2 High-resolution ResNet18 |     0.233001 |       0.19612 |        0.19655 |
| Model C: C5b Multi-seed LightGBM     |     0.162576 |       0.14293 |        0.13985 |
| Equal three-model average            |     0.182347 |       0.16385 |        0.16187 |
| Cross-fitted static blend            |     0.155265 |       0.13854 |        0.13617 |
| Cross-fitted logistic stacking       | **0.139562** |   **0.11713** |    **0.11392** |

## Repository layout

```text
CS4486-Project/
├── artifacts/             # Reproducible folds, metrics, and final predictions
├── data/
│   └── raw/               # Local Kaggle files; ignored by Git
├── notebooks/             # Exploratory data analysis
├── scripts/               # Data download and utility scripts
├── src/
│   ├── model_a/           # Wider SimpleCNN pipeline
│   ├── model_b/           # ResNet18 and input/angle experiments
│   ├── model_c/           # LightGBM C5b pipeline
│   └── ensemble/          # Three-model ensemble
├── tests/                 # Unit and artifact-interface tests
└── environment.yml        # Reproducible Conda environment
```

## Data

Competition: [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge)

Accept the competition rules before downloading the files. Place the extracted files in `data/raw/`. See `data/README.md` for the required names.

From the repository root, KaggleHub can download the competition files with:

```powershell
python scripts\download_data.py
```

A `403` response normally means that the authenticated Kaggle account has not
accepted the competition rules or that the token belongs to another account.

## Environment

```powershell
conda env create -f environment.yml
conda activate iceberg
python -m ipykernel install --user --name iceberg --display-name "Python (iceberg)"
```

The environment uses PyTorch CUDA 12.6 wheels, NVIDIA driver CUDA 12.7.

## Reproduction

Run commands from the repository root.

### Model A

```powershell
python src\model_a\run_a2.py --experiment a2_14 --epochs 10
```

### Model B

```powershell
python src\model_b\run_resnet.py --experiment r1 --epochs 10 --learning-rate 1e-4 --weight-decay 1e-4 --dropout 0.3 --optimizer adamw --augmentation flip --pretrained --high-resolution-stem --tta none --run-tag e2_highres_stem
```

Additional Model B entry points:

```powershell
python src\model_b\run_i1.py --help
python src\model_b\run_angle.py --help
python src\model_b\run_temperature_scaling.py --help
```

### Model C

The C5b result depends on earlier C3 and C4 outputs. Run the complete chain in
this order:

```powershell
python src\model_c\10_lightgbm_c1e_richer_image_gate.py
python src\model_c\C3c_lightgbm_pool10_only_v2.py
python src\model_c\C3d_lightgbm_pool10_seed_robustness_v2.py
python src\model_c\C3e_lightgbm_pool10_multiseed_ensemble_v2.py
python src\model_c\C4a_lightgbm_pool10_foldsafe_pca.py
python src\model_c\C4b_finalize_pca_multiseed_v2.py
python src\model_c\C5a_lightgbm_c4_plus_joint_pca30.py
python src\model_c\C5b_lightgbm_joint_pca_multiseed.py
python src\model_c\13_finalize_model_c.py
python src\model_c\14_make_kaggle_submission_c.py
```

### Final ensemble

The final A, B, and C OOF/Test predictions required by the ensemble script are
included in `artifacts/predictions/`.

```powershell
python src\ensemble\ensemble_three_models.py
```

Expected OOF log losses:

```text
equal average       0.182347204738591
static cross-fit    0.155264616612858
logistic stacking   0.139561519570046
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover augmentation, ResNet output shapes, blending behavior, the
download helper, shared prediction interfaces, submission schemas, and final
ensemble scores.

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss


EPS = 1e-6


def apply_temperature(probabilities, temperature):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)

    logits = logit(probabilities)
    return expit(logits / temperature)


def find_temperature(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    def objective(temperature):
        calibrated = apply_temperature(probabilities, temperature)

        return log_loss(targets, calibrated, labels=[0, 1])

    result = minimize_scalar(objective, bounds=(0.25, 4.0), method="bounded")

    if not result.success:
        raise RuntimeError("Temperature optimization failed.")

    return float(result.x)


def crossfit_temperature(targets, probabilities, folds):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int64)

    calibrated_oof = np.zeros(len(targets), dtype=np.float64)
    fold_records = []

    unique_folds = sorted(np.unique(folds).tolist())

    for fold in unique_folds:
        train_mask = folds != fold
        validation_mask = folds == fold

        temperature = find_temperature(targets[train_mask], probabilities[train_mask])

        calibrated_oof[validation_mask] = apply_temperature(probabilities[validation_mask], temperature)

        raw_fold_loss = log_loss(targets[validation_mask], probabilities[validation_mask], labels=[0, 1])

        calibrated_fold_loss = log_loss(targets[validation_mask], calibrated_oof[validation_mask], labels=[0, 1])

        fold_records.append(
            {
                "fold": fold,
                "temperature": temperature,
                "raw_fold_log_loss": raw_fold_loss,
                "calibrated_fold_log_loss": calibrated_fold_loss,
            }
        )

    return calibrated_oof, fold_records
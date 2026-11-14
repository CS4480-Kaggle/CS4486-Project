from pathlib import Path
import unittest

import pandas as pd
from sklearn.metrics import log_loss


PREDICTION_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "predictions"
)

EXPECTED_OOF_LOG_LOSS = {
    "ensemble_abc_equal_seed42": 0.18234720473859078,
    "ensemble_abc_static_crossfit_seed42": 0.1552646166128576,
    "ensemble_abc_logit_stack_c0p1_seed42": 0.13956151957004556,
}


class TestEnsembleArtifacts(unittest.TestCase):
    def test_recomputed_oof_losses_match_frozen_results(self):
        for run_id, expected_loss in EXPECTED_OOF_LOG_LOSS.items():
            with self.subTest(run_id=run_id):
                oof = pd.read_csv(PREDICTION_DIR / f"oof_{run_id}.csv")
                actual_loss = log_loss(oof["target"], oof["prediction"])
                self.assertAlmostEqual(actual_loss, expected_loss, places=12)


if __name__ == "__main__":
    unittest.main()

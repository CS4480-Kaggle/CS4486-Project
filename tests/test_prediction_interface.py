from pathlib import Path
import unittest

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO_ROOT / "artifacts"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"

MODEL_RUNS = {
    "a": "a2_14_mean3ch_wide_24_48_96_seed42",
    "b": (
        "r1_resnet18_highres_pretrained_image_angle_"
        "lr0p0001_wd0p0001_do0p3_adamw_augflip_"
        "ttanone_10ep_seed42_e2_highres_stem"
    ),
    "c": "lightgbm_c_c5b_c4_plus_joint_pca30_multiseed_42_52_62",
}


class TestPredictionInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared_folds = pd.read_csv(ARTIFACT_DIR / "folds.csv")

    def test_final_model_predictions_follow_shared_interface(self):
        reference = None

        for model_key, run_id in MODEL_RUNS.items():
            with self.subTest(model=model_key):
                oof = pd.read_csv(PREDICTION_DIR / f"oof_{run_id}.csv")
                test = pd.read_csv(PREDICTION_DIR / f"test_{run_id}.csv")

                self.assertEqual(
                    list(oof.columns),
                    ["id", "fold", "target", "prediction", "model_name", "run_id"],
                )
                self.assertEqual(
                    list(test.columns),
                    ["id", "prediction", "model_name", "run_id"],
                )
                self.assertEqual(len(oof), 1604)
                self.assertEqual(len(test), 8424)
                self.assertEqual(oof["id"].nunique(), 1604)
                self.assertEqual(test["id"].nunique(), 8424)
                self.assertTrue(oof["prediction"].between(0.0, 1.0).all())
                self.assertTrue(test["prediction"].between(0.0, 1.0).all())

                aligned = oof[["id", "fold"]].merge(
                    self.shared_folds[["id", "fold"]],
                    on="id",
                    suffixes=("_prediction", "_shared"),
                    validate="one_to_one",
                )
                self.assertTrue(
                    aligned["fold_prediction"].equals(aligned["fold_shared"])
                )

                current = oof[["id", "fold", "target"]].sort_values("id")
                current = current.reset_index(drop=True)
                if reference is None:
                    reference = current
                else:
                    pd.testing.assert_frame_equal(reference, current)

    def test_final_submissions_follow_kaggle_interface(self):
        submission_files = [
            PREDICTION_DIR
            / "submission_a2_14_mean3ch_wide_24_48_96_seed42.csv",
            PREDICTION_DIR
            / (
                "submission_r1_resnet18_highres_pretrained_image_angle_"
                "lr0p0001_wd0p0001_do0p3_adamw_augflip_"
                "ttanone_10ep_seed42_e2_highres_stem.csv"
            ),
            PREDICTION_DIR / "submission_lightgbm_c5b_joint_pca30_multiseed.csv",
            PREDICTION_DIR / "submission_ensemble_abc_equal_seed42.csv",
            PREDICTION_DIR / "submission_ensemble_abc_static_crossfit_seed42.csv",
            PREDICTION_DIR / "submission_ensemble_abc_logit_stack_c0p1_seed42.csv",
        ]

        for path in submission_files:
            with self.subTest(path=path.name):
                submission = pd.read_csv(path)
                self.assertEqual(list(submission.columns), ["id", "is_iceberg"])
                self.assertEqual(len(submission), 8424)
                self.assertEqual(submission["id"].nunique(), 8424)
                self.assertTrue(submission["is_iceberg"].between(0.0, 1.0).all())


if __name__ == "__main__":
    unittest.main()

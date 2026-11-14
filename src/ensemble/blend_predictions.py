from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def blend_probabilities(first, second, first_weight):
    return first_weight * first + (1.0 - first_weight) * second


def search_blend_weight(targets, first, second, step=0.05):
    rows = []

    for first_weight in np.arange(0.0, 1.0 + step / 2.0, step):
        blended = blend_probabilities(first, second, first_weight)
        rows.append({
            "first_weight": round(float(first_weight), 10),
            "second_weight": round(float(1.0 - first_weight), 10),
            "oof_log_loss": log_loss(targets, blended, labels=[0, 1]),
        })

    return pd.DataFrame(rows).sort_values(
        "oof_log_loss",
        ascending=True,
    ).reset_index(drop=True)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Blend two OOF and test prediction runs."
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--weight-a", type=float, default=None)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    repo_root = Path(__file__).resolve().parents[2]
    prediction_dir = repo_root / "artifacts" / "predictions"
    metric_dir = repo_root / "artifacts" / "metrics"

    first_oof = pd.read_csv(prediction_dir / f"oof_{arguments.run_a}.csv")
    second_oof = pd.read_csv(prediction_dir / f"oof_{arguments.run_b}.csv")

    oof = first_oof[["id", "fold", "target", "prediction"]].rename(
        columns={"prediction": "first_prediction"}
    ).merge(
        second_oof[["id", "prediction"]].rename(
            columns={"prediction": "second_prediction"}
        ),
        on="id",
        validate="one_to_one",
    )

    if arguments.weight_a is None:
        weight_table = search_blend_weight(
            oof["target"].to_numpy(),
            oof["first_prediction"].to_numpy(),
            oof["second_prediction"].to_numpy(),
            step=arguments.step,
        )
        first_weight = float(weight_table.loc[0, "first_weight"])
    else:
        first_weight = arguments.weight_a
        blended_oof = blend_probabilities(
            oof["first_prediction"].to_numpy(),
            oof["second_prediction"].to_numpy(),
            first_weight,
        )
        weight_table = pd.DataFrame([{
            "first_weight": first_weight,
            "second_weight": 1.0 - first_weight,
            "oof_log_loss": log_loss(
                oof["target"],
                blended_oof,
                labels=[0, 1],
            ),
        }])

    second_weight = 1.0 - first_weight
    oof_predictions = blend_probabilities(
        oof["first_prediction"].to_numpy(),
        oof["second_prediction"].to_numpy(),
        first_weight,
    )
    oof_log_loss = log_loss(oof["target"], oof_predictions, labels=[0, 1])

    first_test = pd.read_csv(
        prediction_dir / f"test_{arguments.run_a}.csv"
    )[["id", "prediction"]].rename(columns={"prediction": "first_prediction"})
    second_test = pd.read_csv(
        prediction_dir / f"test_{arguments.run_b}.csv"
    )[["id", "prediction"]].rename(columns={"prediction": "second_prediction"})
    test = first_test.merge(second_test, on="id", validate="one_to_one")
    test_predictions = blend_probabilities(
        test["first_prediction"].to_numpy(),
        test["second_prediction"].to_numpy(),
        first_weight,
    )

    oof_output = pd.DataFrame({
        "id": oof["id"],
        "fold": oof["fold"],
        "target": oof["target"],
        "prediction": oof_predictions,
        "model_name": "ensemble",
        "run_id": arguments.output_id,
    })
    test_output = pd.DataFrame({
        "id": test["id"],
        "prediction": test_predictions,
        "model_name": "ensemble",
        "run_id": arguments.output_id,
    })
    submission = test_output[["id", "prediction"]].rename(
        columns={"prediction": "is_iceberg"}
    )
    summary = pd.DataFrame([{
        "run_id": arguments.output_id,
        "model_name": "ensemble",
        "run_a": arguments.run_a,
        "run_b": arguments.run_b,
        "weight_a": first_weight,
        "weight_b": second_weight,
        "overall_oof_log_loss": oof_log_loss,
    }])

    oof_output.to_csv(
        prediction_dir / f"oof_{arguments.output_id}.csv",
        index=False,
    )
    test_output.to_csv(
        prediction_dir / f"test_{arguments.output_id}.csv",
        index=False,
    )
    submission.to_csv(
        prediction_dir / f"submission_{arguments.output_id}.csv",
        index=False,
    )
    weight_table.to_csv(
        metric_dir / f"blend_weights_{arguments.output_id}.csv",
        index=False,
    )
    summary.to_csv(
        metric_dir / f"summary_{arguments.output_id}.csv",
        index=False,
    )

    print(weight_table)
    print()
    print("Selected first model weight:", first_weight)
    print("Selected second model weight:", second_weight)
    print("Ensemble OOF Log Loss:", oof_log_loss)
    print(
        "Submission saved to:",
        prediction_dir / f"submission_{arguments.output_id}.csv",
    )


if __name__ == "__main__":
    main()

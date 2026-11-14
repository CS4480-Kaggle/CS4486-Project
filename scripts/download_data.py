from argparse import ArgumentParser
from pathlib import Path

import kagglehub


COMPETITION = "statoil-iceberg-classifier-challenge"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPO_ROOT / "data" / "raw"


def download_competition_data(destination, force_download=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    downloaded_path = kagglehub.competition_download(
        COMPETITION,
        output_dir=str(destination),
        force_download=force_download,
    )
    return Path(downloaded_path)


def parse_arguments():
    parser = ArgumentParser(
        description="Download the Statoil iceberg competition data with KaggleHub."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Directory that receives the downloaded competition files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the competition files again even if they are cached.",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    downloaded_path = download_competition_data(
        destination=arguments.destination,
        force_download=arguments.force_download,
    )
    print(f"Competition data downloaded to: {downloaded_path}")


if __name__ == "__main__":
    main()

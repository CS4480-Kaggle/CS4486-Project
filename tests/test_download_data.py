from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from download_data import COMPETITION, download_competition_data


class TestDownloadData(unittest.TestCase):
    def test_downloads_competition_to_requested_directory(self):
        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "raw"
            downloaded_path = destination / "statoil-data"

            with patch(
                "download_data.kagglehub.competition_download",
                return_value=str(downloaded_path),
            ) as competition_download:
                actual = download_competition_data(destination)

            self.assertEqual(actual, downloaded_path)
            self.assertTrue(destination.is_dir())
            competition_download.assert_called_once_with(
                COMPETITION,
                output_dir=str(destination),
                force_download=False,
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ingestion import bootstrap


class CloudBootstrapTests(unittest.TestCase):
    def test_detects_chroma_persistence_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(bootstrap.has_persisted_vectorstore(temp_dir))
            Path(temp_dir, "chroma.sqlite3").touch()
            self.assertTrue(bootstrap.has_persisted_vectorstore(temp_dir))

    @patch("src.ingestion.bootstrap.run_pipeline", return_value={"indexed_chunks": 3})
    def test_builds_from_tracked_sample_without_scrapers(self, run_pipeline):
        self.assertEqual(bootstrap.build_sample_vectorstore(), 3)
        run_pipeline.assert_called_once_with(run_scrapers=False, target_dir="./data/raw_sample")


if __name__ == "__main__":
    unittest.main()
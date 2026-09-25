import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.operations import atomic_json
from training.scripts.server_job import status


class JobStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        atomic_json(self.root / "status.json", {
            "status": "running", "supervisorPid": 101, "supervisorIdentity": "start-a",
            "childPid": 102, "childIdentity": "start-b",
        })

    def tearDown(self):
        self.temporary.cleanup()

    def test_live_supervisor(self):
        with patch("training.scripts.server_job.process_identity", return_value="start-a"):
            self.assertEqual(status(self.root)["status"], "running")

    def test_gone_supervisor_and_child(self):
        with patch("training.scripts.server_job.process_identity", return_value=None):
            self.assertEqual(status(self.root)["status"], "interrupted")

    def test_orphan_child_is_not_resumable(self):
        with patch("training.scripts.server_job.process_identity", side_effect=[None, "start-b"]):
            self.assertEqual(status(self.root)["status"], "orphaned")

    def test_reused_pid_is_not_our_job(self):
        with patch("training.scripts.server_job.process_identity", return_value="another-process"):
            self.assertEqual(status(self.root)["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()

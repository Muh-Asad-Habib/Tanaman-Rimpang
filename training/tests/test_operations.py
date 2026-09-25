import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from training.common import digest
from training.operations import atomic_json, extract_artifacts, identifier, package_artifacts, relative_path


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "run-v1"
        self.run.mkdir()
        atomic_json(self.run / "report.json", {"status": "blocked", "reason": "human-review"})

    def tearDown(self):
        self.temporary.cleanup()

    def test_identifiers_and_paths(self):
        self.assertEqual(identifier("run-v1"), "run-v1")
        for value in ("../bad", "-bad", "bad id", "..", "bad;command"):
            with self.assertRaises(ValueError):
                identifier(value)
        for value in ("../escape", "C:/private", "/absolute", "folder\\image", "a/../b", "a//b"):
            with self.assertRaises(ValueError):
                relative_path(value)

    def test_incomplete_is_explicit_and_roundtrips(self):
        with self.assertRaises(ValueError):
            package_artifacts(self.run, self.root / "transfers")
        result = package_artifacts(self.run, self.root / "transfers", allow_incomplete=True)
        self.assertEqual(result, package_artifacts(self.run, self.root / "transfers", allow_incomplete=True))
        destination = self.root / "received"
        manifest = extract_artifacts(result["archive"], destination, result["sha256"])
        self.assertEqual(manifest["status"], "incomplete")
        self.assertEqual(json.loads((destination / "report.json").read_text())["reason"], "human-review")
        with self.assertRaises(ValueError):
            extract_artifacts(result["archive"], destination, result["sha256"])

    def test_bad_archive_hash_does_not_create_destination(self):
        result = package_artifacts(self.run, self.root / "transfers", allow_incomplete=True)
        destination = self.root / "received"
        with self.assertRaises(ValueError):
            extract_artifacts(result["archive"], destination, "0" * 64)
        self.assertFalse(destination.exists())

    def test_rejects_path_traversal_and_links(self):
        for name, kind in (("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)):
            archive = self.root / f"bad-{kind.decode()}.tar"
            with tarfile.open(archive, "w") as bundle:
                member = tarfile.TarInfo(name)
                member.type = kind
                if kind == tarfile.SYMTYPE:
                    member.linkname = "../escape"
                else:
                    member.size = 3
                bundle.addfile(member, io.BytesIO(b"bad") if kind == tarfile.REGTYPE else None)
            with self.assertRaises(ValueError):
                extract_artifacts(archive, self.root / "received", digest(archive))
        self.assertFalse((self.root / "received").exists())

    def test_refuses_to_snapshot_live_job(self):
        atomic_json(self.run / "jobs" / "train" / "status.json", {"status": "running"})
        with self.assertRaises(ValueError):
            package_artifacts(self.run, self.root / "transfers", allow_incomplete=True)

    def test_atomic_metadata_replacement(self):
        target = self.root / "nested" / "state.json"
        atomic_json(target, {"attempt": 1})
        atomic_json(target, {"attempt": 2})
        self.assertEqual(json.loads(target.read_text()), {"attempt": 2})
        self.assertEqual([file.name for file in target.parent.iterdir()], ["state.json"])


if __name__ == "__main__":
    unittest.main()

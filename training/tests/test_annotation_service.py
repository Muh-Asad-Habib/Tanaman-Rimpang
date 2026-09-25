import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.common import read_json, write_json
from training.scripts.annotation_service import import_tasks, task_identity


class AnnotationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tasks = [{"data": {"image": "/data/local-files/?d=jahe/a.jpg", "path": "jahe/a.jpg"}}]
        write_json(self.root / "tasks.json", self.tasks)
        (self.root / "config.xml").write_text("<View />")
        self.args = SimpleNamespace(service=self.root, port=8087, tasks=self.root / "tasks.json",
                                    label_config=self.root / "config.xml", project_state=self.root / "project.json",
                                    title="Review")

    def tearDown(self):
        self.temporary.cleanup()

    def test_import_has_verified_inventory_and_is_idempotent(self):
        with patch("training.scripts.annotation_service.api", side_effect=[
            {"id": 7}, {"task_count": 1}, self.tasks,
        ]) as call, contextlib.redirect_stdout(io.StringIO()):
            import_tasks(self.args)
        self.assertEqual(call.call_count, 3)
        state = read_json(self.args.project_state)
        self.assertEqual(state["status"], "awaiting-human-review")
        with patch("training.scripts.annotation_service.api", side_effect=[self.tasks, self.tasks]) as call:
            with contextlib.redirect_stdout(io.StringIO()):
                import_tasks(self.args)
        self.assertEqual(call.call_count, 2)

    def test_export_identity_ignores_annotation_progress(self):
        self.assertEqual(task_identity(self.tasks[0]),
                         task_identity({**self.tasks[0], "id": 123, "annotations": [{"result": []}]}))

    def test_duplicate_input_rejected_before_creating_project(self):
        write_json(self.args.tasks, self.tasks + self.tasks)
        with patch("training.scripts.annotation_service.api") as call:
            with self.assertRaises(ValueError):
                import_tasks(self.args)
            call.assert_not_called()

    def test_stale_input_cannot_mix_with_existing_project(self):
        with patch("training.scripts.annotation_service.api", side_effect=[{"id": 7}, {}, self.tasks]):
            with contextlib.redirect_stdout(io.StringIO()):
                import_tasks(self.args)
        self.args.label_config.write_text("<View><Header value='changed' /></View>")
        with patch("training.scripts.annotation_service.api") as call:
            with self.assertRaises(ValueError):
                import_tasks(self.args)
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()

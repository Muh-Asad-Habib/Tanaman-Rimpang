import contextlib
import copy
import io
import json
import shutil
import sys
import unittest
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from training.annotations import (
    import_annotation_tasks,
    prepare_annotation_tasks,
    publish_annotation_files,
    read_annotation_json,
)
from training.common import digest, read_json, taxonomy, validate_schema, write_json
from training.dataset import assign_splits, audit_manifest
from training.scripts.import_annotations import main as import_main
from training.scripts.prepare_annotations import main as prepare_main


def control(name, value, kind="choices"):
    return {"id": f"global-{name}", "from_name": name, "to_name": "image",
            "type": kind, "origin": "manual", "value": value}


def rectangle(slug="jahe", result_id="box-1", width=100, height=80):
    return {
        "id": result_id, "from_name": "bbox", "to_name": "image", "type": "rectanglelabels",
        "origin": "manual", "original_width": width, "original_height": height, "image_rotation": 0,
        "value": {"x": 10, "y": 20, "width": 50, "height": 40, "rotation": 0, "rectanglelabels": [slug]},
    }


def completed_task(task, task_id, decision="include", group="session-common"):
    task = copy.deepcopy(task)
    results = [
        control("decision", {"choices": [decision]}),
        control("reviewed", {"choices": ["reviewed"]}),
    ]
    if decision != "exclude":
        results.append(control("groupId", {"text": [group]}, "textarea"))
    if decision == "include":
        results.append(rectangle(width=task["data"]["width"], height=task["data"]["height"]))
    if decision == "exclude":
        results.append(control("excludeReason", {"text": ["Sumber/taksonomi belum dapat dipastikan."]}, "textarea"))
    task.update(id=task_id, is_labeled=True, drafts=[], predictions=[], annotations=[{
        "id": 1000 + task_id, "task": task_id, "completed_by": 7, "was_cancelled": False, "result": results,
    }])
    return task


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / f".annotation-test-{uuid.uuid4().hex}"
        self.images = self.root / "images"
        self.images.mkdir(parents=True)
        self.rows = []
        for index, relative in enumerate(("a/ginger #1.png", "b/non target.png", "c/uncertain.png")):
            path = self.images.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 80), (index * 70, 30, 50)).save(path)
            self.rows.append({"path": relative, "sourceId": f"drive:fixture-{index}", "proposedClassId": index,
                              "sha256": digest(path), "width": 100, "height": 80})
        self.rows[0]["license"] = "synthetic unit-test fixture"
        self.inventory = {"schemaVersion": 1, "labelsVersion": 1, "images": self.rows}
        self.tasks, self.config, self.sidecar = prepare_annotation_tasks(self.inventory, self.images)
        self.export = [completed_task(task, index + 1, decision) for index, (task, decision) in enumerate(
            zip(self.tasks, ("include", "negative", "exclude"))
        )]

    def tearDown(self):
        shutil.rmtree(self.root)

    def results(self, export=None, task_index=0):
        return (self.export if export is None else export)[task_index]["annotations"][0]["result"]

    def result(self, name, export=None, task_index=0):
        return next(result for result in self.results(export, task_index) if result["from_name"] == name)

    def import_value(self, export=None, sidecar=None):
        return import_annotation_tasks(self.export if export is None else export,
                                       self.sidecar if sidecar is None else sidecar, self.images)

    def assert_blocked(self, export, expected_status=None):
        manifest, report = self.import_value(export)
        self.assertIsNone(manifest)
        self.assertNotEqual(report["status"], "ready")
        self.assertTrue(report["errors"] or report["pending"])
        if expected_status:
            self.assertEqual(report["status"], expected_status)
        return report

    def test_normalized_inventory_adapter_dedupes_exact_images(self):
        duplicate = self.images / "a" / "copy.png"
        shutil.copyfile(self.images / "a" / "ginger #1.png", duplicate)
        rows = [{"path": row["path"], "sha256": row["sha256"], "width": 100, "height": 80,
                 "driveId": f"id-{index}", "originalPath": f"orig/{index}.jpg",
                 "classProposal": {"classId": row["proposedClassId"], "slug": "x", "status": "pending"}}
                for index, row in enumerate(self.rows)]
        rows.append({**rows[0], "path": "a/copy.png", "driveId": "id-dup",
                     "classProposal": {"classId": 4, "slug": "x", "status": "pending"}})
        document = {"schemaVersion": 1, "kind": "normalized-image-inventory", "status": "complete",
                    "labelsVersion": 1, "images": rows}
        output = self.root / "tasks-v1"
        with contextlib.redirect_stdout(io.StringIO()):
            write_json(self.root / "inventory.json", document)
            self.assertEqual(prepare_main(["--inventory", str(self.root / "inventory.json"),
                                           "--image-root", str(self.images), "--output", str(output)]), 0)
        tasks = read_json(output / "tasks.json")
        report = read_json(output / "duplicates.json")
        self.assertEqual(len(tasks), 3)
        self.assertEqual(report["excludedAliases"], 1)
        self.assertEqual(report["clusters"][0]["representative"], "a/copy.png")
        self.assertTrue(report["clusters"][0]["classConflict"])
        self.assertTrue(all(task["data"]["sourceId"].startswith("drive:") for task in tasks))
        with self.assertRaises(ValueError):
            from training.annotations import adapt_normalized_inventory
            adapt_normalized_inventory({**document, "status": "partial"})

    def test_preparation_is_stable_private_and_unannotated(self):
        reverse = {**self.inventory, "images": list(reversed(self.rows))}
        self.assertEqual(prepare_annotation_tasks(reverse, self.images), (self.tasks, self.config, self.sidecar))
        for task, row in zip(self.tasks, self.rows):
            self.assertEqual(set(task), {"data"})
            self.assertEqual(task["data"]["sha256"], row["sha256"])
            self.assertNotIn("groupId", task["data"])
        self.assertEqual(self.tasks[0]["data"]["image"], "/data/local-files/?d=a/ginger%20%231.png")
        self.assertEqual(len(list(self.images.rglob("*.png"))), 3)
        xml = ET.fromstring(self.config)
        labels = xml.findall("./RectangleLabels/Label")
        self.assertEqual([item.attrib["value"] for item in labels], [row["slug"] for row in taxonomy()["labels"]])
        self.assertEqual([int(item.attrib["model_index"]) for item in labels], list(range(10)))
        self.assertEqual(xml.find("./RectangleLabels").attrib["canRotate"], "false")
        self.assertEqual(xml.find("./Image").attrib["rotateControl"], "false")
        self.assertEqual(xml.find("./Choices[@name='reviewed']").attrib["required"], "true")
        self.assertEqual(xml.find("./Choices[@name='decision']").attrib["required"], "true")
        for field in ("groupId", "excludeReason"):
            self.assertNotIn("value", xml.find(f"./TextArea[@name='{field}']").attrib)

    def test_inventory_requires_explicit_contract_and_valid_types(self):
        mutations = [
            lambda inv: inv.pop("labelsVersion"),
            lambda inv: inv.update(schemaVersion=True),
            lambda inv: inv.update(labelsVersion=2),
            lambda inv: inv.update(provenance={}),
            lambda inv: inv.update(images=[]),
            lambda inv: inv["images"][0].pop("sourceId"),
            lambda inv: inv["images"][0].update(sourceId=" \t "),
            lambda inv: inv["images"][0].update(proposedClassId=True),
            lambda inv: inv["images"][0].update(proposedClassId=10),
            lambda inv: inv["images"][0].update(proposedClassId=-1),
            lambda inv: inv["images"][0].update(proposedClassId="0"),
            lambda inv: inv["images"][0].update(width=15),
            lambda inv: inv["images"][0].update(height=80.0),
            lambda inv: inv["images"][0].update(sha256="A" * 64),
            lambda inv: inv["images"][0].update(sha256="bad"),
            lambda inv: inv["images"][0].update(hash=inv["images"][0].pop("sha256")),
            lambda inv: inv["images"][0].update(groupId="invented"),
            lambda inv: inv["images"][0].update(license=[]),
            lambda inv: inv["images"].append(copy.deepcopy(inv["images"][0])),
            lambda inv: inv["images"][1].update(sha256=inv["images"][0]["sha256"]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                candidate = copy.deepcopy(self.inventory)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    prepare_annotation_tasks(candidate, self.images)

    def test_no_path_traversal_absolute_or_missing_images(self):
        for path in ("../outside.png", "a/../../outside.png", "/a.png", "\\outside.png",
                     "C:\\secret.png", "a/./image.png", "a//image.png", "missing.png", "a/\x00image.png"):
            with self.subTest(path=path):
                candidate = copy.deepcopy(self.inventory)
                candidate["images"][0]["path"] = path
                with self.assertRaises(ValueError):
                    prepare_annotation_tasks(candidate, self.images)

    def test_windows_separators_normalize_to_portable_identity(self):
        candidate = copy.deepcopy(self.inventory)
        candidate["images"][0]["path"] = candidate["images"][0]["path"].replace("/", "\\")
        self.assertEqual(prepare_annotation_tasks(candidate, self.images), (self.tasks, self.config, self.sidecar))

    def test_out_of_root_symlink_is_rejected(self):
        outside = self.root / "outside.png"
        outside.write_bytes((self.images / self.rows[0]["path"]).read_bytes())
        link = self.images / "linked.png"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Creating symlinks requires privileges on this Windows host.")
        candidate = copy.deepcopy(self.inventory)
        candidate["images"][0]["path"] = "linked.png"
        with self.assertRaises(ValueError):
            prepare_annotation_tasks(candidate, self.images)

    def test_actual_hash_dimensions_and_orientation_must_match(self):
        for changes in ({"sha256": "0" * 64}, {"width": 101}, {"height": 81}):
            with self.subTest(changes=changes):
                candidate = copy.deepcopy(self.inventory)
                candidate["images"][0].update(changes)
                with self.assertRaises(ValueError):
                    prepare_annotation_tasks(candidate, self.images)
        path = self.images / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (100, 80), "red").save(path, exif=exif)
        candidate = copy.deepcopy(self.inventory)
        candidate["images"][0].update(path="oriented.jpg", sha256=digest(path))
        with self.assertRaisesRegex(ValueError, "orientation"):
            prepare_annotation_tasks(candidate, self.images)

    def test_reviewed_positive_negative_excluded_roundtrip(self):
        self.result("bbox")["value"]["rectanglelabels"] = ["temulawak"]
        manifest, report = self.import_value()
        validate_schema(manifest, "dataset-manifest.schema.json")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["counts"], {"expected": 3, "exported": 3, "accepted": 2, "included": 1,
                                          "negative": 1, "excluded": 1, "pending": 0, "errorTasks": 0, "errors": 0})
        self.assertEqual(manifest["images"][0]["objects"], [{"classId": 9, "bbox": [0.1, 0.2, 0.5, 0.4]}])
        self.assertEqual(manifest["images"][1]["objects"], [])
        self.assertEqual(manifest["images"][0]["license"], self.rows[0]["license"])
        self.assertEqual(manifest["images"][0]["sha256"], self.rows[0]["sha256"])
        self.assertEqual({row["groupId"] for row in manifest["images"]}, {"session-common"})
        self.assertEqual(report["excluded"][0]["path"], self.rows[2]["path"])
        self.assertTrue(report["excluded"][0]["reason"])
        self.assertEqual(report["accepted"][0]["reviewerIds"], [7])
        records, audit = audit_manifest(manifest, self.images)
        self.assertEqual(audit["negativeImages"], 1)
        with self.assertRaisesRegex(ValueError, "three independent groups"):
            assign_splits(records)

    def test_full_frame_box_is_only_accepted_after_manual_review(self):
        self.result("bbox")["value"].update(x=0, y=0, width=100, height=100)
        manifest, _ = self.import_value()
        self.assertEqual(manifest["images"][0]["objects"][0]["bbox"], [0, 0, 1, 1])
        self.results().remove(self.result("reviewed"))
        self.assert_blocked(self.export, "incomplete")

    def test_predictions_cancelled_drafts_and_unlabeled_are_not_reviews(self):
        mutations = [
            lambda task: task.update(predictions=task.pop("annotations")),
            lambda task: task.update(annotations=[], drafts=[{"result": []}]),
            lambda task: task["annotations"][0].update(was_cancelled=True),
            lambda task: task.update(is_labeled=False),
            lambda task: task.update(was_cancelled=True),
            lambda task: task.update(skipped=True),
            lambda task: task.update(is_skipped=True),
            lambda task: task.update(draft=True),
            lambda task: task.update(is_draft=True),
            lambda task: task.update(drafts=[{"result": []}]),
            lambda task: task["annotations"][0].update(is_draft=True),
            lambda task: task["annotations"][0].update(skipped=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                candidate = copy.deepcopy(self.export)
                mutate(candidate[0])
                report = self.assert_blocked(candidate, "incomplete")
                self.assertEqual(report["counts"]["pending"], 1)

    def test_export_without_optional_is_labeled_and_with_numeric_string_ids(self):
        self.export[0].pop("is_labeled")
        self.export[0]["id"] = "1"
        self.export[0]["annotations"][0].update(id="1001", task="1", completed_by={"id": 7})
        self.export[0]["predictions"] = [{"result": [rectangle("kencur")]}]
        manifest, report = self.import_value()
        self.assertIsNotNone(manifest)
        self.assertEqual(report["accepted"][0]["annotationIds"], [1001])

    def test_completion_metadata_cannot_be_missing_or_malformed(self):
        mutations = [
            lambda task: task["annotations"][0].pop("was_cancelled"),
            lambda task: task["annotations"][0].update(was_cancelled="false"),
            lambda task: task.update(is_labeled="true"),
            lambda task: task.update(drafts={}),
            lambda task: task.update(annotations={}),
            lambda task: task["annotations"][0].update(completed_by=None),
            lambda task: task["annotations"][0].update(completed_by=True),
            lambda task: task["annotations"][0].update(task=99),
            lambda task: task["annotations"][0].pop("id"),
            lambda task: task["annotations"][0].update(result={}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                candidate = copy.deepcopy(self.export)
                mutate(candidate[0])
                self.assert_blocked(candidate, "invalid")

    def test_explicit_review_decision_and_group_are_required(self):
        for field in ("reviewed", "decision", "groupId"):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.export)
                candidate[0]["annotations"][0]["ground_truth"] = True
                self.results(candidate).remove(self.result(field, candidate))
                self.assert_blocked(candidate, "incomplete")
        for value in ([], [""], ["   "]):
            candidate = copy.deepcopy(self.export)
            self.result("groupId", candidate)["value"]["text"] = value
            self.assert_blocked(candidate, "incomplete")
        self.result("reviewed")["origin"] = "prediction"
        self.assert_blocked(self.export, "incomplete")

    def test_decision_semantics_are_not_inferred_from_empty_objects(self):
        candidate = copy.deepcopy(self.export)
        self.results(candidate).remove(self.result("bbox", candidate))
        self.assert_blocked(candidate, "incomplete")
        candidate = copy.deepcopy(self.export)
        self.result("decision", candidate)["value"]["choices"] = ["negative"]
        self.assert_blocked(candidate, "invalid")
        candidate = copy.deepcopy(self.export)
        self.results(candidate, 2).remove(self.result("excludeReason", candidate, 2))
        self.assert_blocked(candidate, "incomplete")
        candidate = copy.deepcopy(self.export)
        self.results(candidate, 2).append(rectangle())
        self.assert_blocked(candidate, "invalid")
        candidate = [completed_task(task, index + 1, "exclude") for index, task in enumerate(self.tasks)]
        report = self.assert_blocked(candidate, "invalid")
        self.assertEqual(report["counts"]["excluded"], 3)

    def test_global_controls_cannot_be_per_region_or_ambiguous(self):
        mutations = [
            lambda results: results.append(copy.deepcopy(results[0])),
            lambda results: results[0]["value"].update(choices=["include", "negative"]),
            lambda results: results[0]["value"].update(choices="include"),
            lambda results: results[0].update(from_name="different-config"),
            lambda results: results[0].update(to_name="different-image"),
            lambda results: results[0].update(type="rectanglelabels"),
            lambda results: results[0].update(parentID="box-1"),
            lambda results: results[0].pop("id"),
            lambda results: results[2]["value"].update(text=["group-a", "group-b"]),
            lambda results: results[2]["value"].update(text=["group\nper-file"]),
            lambda results: results.append(control("excludeReason", {"text": ["stale exclusion"]}, "textarea")),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                candidate = copy.deepcopy(self.export)
                mutate(self.results(candidate))
                self.assert_blocked(candidate, "invalid")

    def test_all_taxonomy_ids_and_downstream_group_disjoint_splits(self):
        rows = []
        for group in range(3):
            for class_id in range(10):
                path = self.images / f"group-{group}-class-{class_id}.png"
                Image.new("RGB", (100, 80), (class_id * 20, group * 70, 99)).save(path)
                rows.append({"path": path.name, "sourceId": "synthetic-shared-source", "proposedClassId": class_id,
                             "sha256": digest(path), "width": 100, "height": 80})
        inventory = {"schemaVersion": 1, "labelsVersion": 1, "images": rows}
        tasks, _, sidecar = prepare_annotation_tasks(inventory, self.images)
        export = []
        for index, task in enumerate(tasks):
            group = task["data"]["path"].split("-")[1]
            completed = completed_task(task, index + 1, group=f"reviewed-session-{group}")
            box = next(result for result in completed["annotations"][0]["result"] if result["type"] == "rectanglelabels")
            box["value"]["rectanglelabels"] = [taxonomy()["labels"][task["data"]["proposedClassId"]]["slug"]]
            export.append(completed)
        manifest, report = import_annotation_tasks(export, sidecar, self.images)
        self.assertEqual(report["status"], "ready")
        split = assign_splits(manifest["images"])
        for name in ("train", "valid", "test"):
            self.assertEqual({obj["classId"] for row in split if row["split"] == name for obj in row["objects"]}, set(range(10)))

    def test_box_numbers_bounds_dimensions_and_rotations_are_validated(self):
        value_changes = [
            {"x": -1}, {"x": 90}, {"x": "10"}, {"x": True}, {"x": float("nan")},
            {"width": 0}, {"height": -1}, {"width": float("inf")}, {"height": 101},
            {"height": 10 ** 1000}, {"rotation": 90}, {"rotation": "0"},
            {"rectanglelabels": ["Jahe"]}, {"rectanglelabels": ["unrecognized"]},
            {"rectanglelabels": ["jahe", "kencur"]}, {"rectanglelabels": []},
        ]
        for changes in value_changes:
            with self.subTest(changes=str(changes)[:100]):
                candidate = copy.deepcopy(self.export)
                self.result("bbox", candidate)["value"].update(changes)
                self.assert_blocked(candidate, "invalid")
        for changes in ({"original_width": 80}, {"original_height": 100}, {"image_rotation": 90},
                        {"image_rotation": False}, {"original_width": None}):
            with self.subTest(changes=changes):
                candidate = copy.deepcopy(self.export)
                self.result("bbox", candidate).update(changes)
                self.assert_blocked(candidate, "invalid")
        candidate = copy.deepcopy(self.export)
        self.result("bbox", candidate).pop("image_rotation")
        self.assert_blocked(candidate, "invalid")
        self.result("bbox")["value"].pop("rotation")
        self.assertIsNotNone(self.import_value()[0])

    def test_multiple_instances_are_preserved_but_duplicate_boxes_rejected(self):
        other = rectangle("kencur", "box-2")
        other["value"].update(x=65, y=5, width=20, height=60)
        self.results().append(other)
        manifest, _ = self.import_value()
        self.assertEqual(len(manifest["images"][0]["objects"]), 2)
        duplicate = copy.deepcopy(other)
        duplicate["id"] = "box-3"
        self.results().append(duplicate)
        self.assert_blocked(self.export, "invalid")

    def test_identical_completed_reviews_are_deterministic_conflicts_rejected(self):
        second = copy.deepcopy(self.export[0]["annotations"][0])
        second.update(id=2001, completed_by=8)
        second["result"].reverse()
        self.export[0]["annotations"].append(second)
        manifest, report = self.import_value()
        self.assertIsNotNone(manifest)
        self.assertEqual(report["accepted"][0]["annotationIds"], [1001, 2001])
        self.export[0]["annotations"].reverse()
        self.assertEqual(self.import_value(), (manifest, report))
        next(result for result in second["result"] if result["from_name"] == "groupId")["value"]["text"] = ["different-session"]
        self.assert_blocked(self.export, "invalid")

    def test_a_second_unreviewed_completion_is_not_ignored(self):
        second = copy.deepcopy(self.export[0]["annotations"][0])
        second["id"] = 2001
        second["result"] = [result for result in second["result"] if result["from_name"] != "reviewed"]
        self.export[0]["annotations"].append(second)
        self.assert_blocked(self.export, "incomplete")

    def test_historical_cancelled_annotation_is_never_used(self):
        cancelled = copy.deepcopy(self.export[0]["annotations"][0])
        cancelled.update(id=2001, was_cancelled=True, result=[])
        self.export[0]["annotations"].insert(0, cancelled)
        manifest, report = self.import_value()
        self.assertIsNotNone(manifest)
        self.assertEqual(report["accepted"][0]["annotationIds"], [1001])

    def test_inconsistent_group_spelling_blocks_publication(self):
        self.result("groupId", task_index=1)["value"]["text"] = ["SESSION-COMMON"]
        self.assert_blocked(self.export, "invalid")
        self.result("groupId", task_index=1)["value"]["text"] = [" session-common "]
        manifest, _ = self.import_value()
        self.assertEqual(manifest["images"][1]["groupId"], "session-common")

    def test_task_data_tampering_is_not_trusted(self):
        mutations = [
            {"image": "https://public.example/image.jpg"}, {"image": "/data/local-files/?d=../secret"},
            {"path": "../secret"}, {"sourceId": "altered"}, {"sha256": "0" * 64},
            {"proposedClassId": 5}, {"proposedClassId": False}, {"proposedClass": "validated"},
            {"width": 101}, {"height": 81}, {"bundleId": "0" * 64}, {"extra": "untrusted"},
        ]
        for changes in mutations:
            with self.subTest(changes=changes):
                candidate = copy.deepcopy(self.export)
                candidate[0]["data"].update(changes)
                self.assert_blocked(candidate, "invalid")

    def test_sidecar_integrity_and_taxonomy_are_checked(self):
        for field in ("bundleId", "taxonomySha256", "configSha256", "inventorySha256"):
            with self.subTest(field=field):
                sidecar = copy.deepcopy(self.sidecar)
                sidecar[field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "Sidecar"):
                    self.import_value(sidecar=sidecar)
        sidecar = copy.deepcopy(self.sidecar)
        sidecar["tasks"][0]["path"] = "../secret.png"
        with self.assertRaises(ValueError):
            self.import_value(sidecar=sidecar)
        changed = taxonomy()
        changed["labels"].reverse()
        with patch("training.annotations.taxonomy", return_value=changed):
            with self.assertRaises(ValueError):
                self.import_value()

    def test_missing_unknown_and_duplicate_identities_are_explicit(self):
        report = self.assert_blocked(self.export[1:], "incomplete")
        self.assertEqual(report["pending"][0]["taskKey"], self.tasks[0]["data"]["taskKey"])
        for mutate in (
            lambda tasks: tasks[0].pop("id"),
            lambda tasks: tasks[0]["data"].pop("taskKey"),
            lambda tasks: tasks[0]["data"].update(taskKey="unknown"),
            lambda tasks: tasks.append(copy.deepcopy(tasks[0])),
            lambda tasks: tasks[1].update(id=tasks[0]["id"]),
            lambda tasks: tasks[1]["annotations"][0].update(id=tasks[0]["annotations"][0]["id"]),
            lambda tasks: tasks[0]["annotations"].append(copy.deepcopy(tasks[0]["annotations"][0])),
        ):
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(self.export)
                mutate(candidate)
                self.assert_blocked(candidate, "invalid")
        with self.assertRaises(ValueError):
            self.import_value(export={"tasks": self.export})

    def test_import_rechecks_all_source_images_including_exclusions(self):
        path = self.images / self.rows[2]["path"]
        Image.new("RGB", (100, 80), "blue").save(path)
        report = self.assert_blocked(self.export, "invalid")
        self.assertEqual(report["counts"]["excluded"], 0)
        self.assertIn("checksum mismatch", report["errors"][0]["reason"])
        path.unlink()
        self.assert_blocked(self.export, "invalid")

    def test_read_json_rejects_duplicate_keys_and_nonfinite_values(self):
        path = self.root / "invalid.json"
        for content in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_annotation_json(path)
        path.write_text('{"key": "value"}', encoding="utf-8-sig")
        self.assertEqual(read_annotation_json(path), {"key": "value"})

    def test_cli_ready_and_pending_outputs_are_persistent_and_never_overwritten(self):
        inventory, exported = self.root / "inventory.json", self.root / "export.json"
        prepared, ready, pending = self.root / "tasks", self.root / "ready", self.root / "pending"
        write_json(inventory, self.inventory)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_main(["--inventory", str(inventory), "--image-root", str(self.images),
                                          "--output", str(prepared)]), 0)
        self.assertEqual(read_json(prepared / "sidecar.json"), self.sidecar)
        write_json(exported, self.export)
        common = ["--export", str(exported), "--sidecar", str(prepared / "sidecar.json"),
                  "--image-root", str(self.images)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(import_main([*common, "--output", str(ready)]), 0)
        validate_schema(read_json(ready / "manifest.json"), "dataset-manifest.schema.json")
        original = (ready / "manifest.json").read_bytes()
        write_json(exported, self.export[1:])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(import_main([*common, "--output", str(pending)]), 2)
        self.assertEqual(read_json(pending / "review-report.json")["status"], "incomplete")
        self.assertFalse((pending / "manifest.json").exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            import_main([*common, "--output", str(ready)])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((ready / "manifest.json").read_bytes(), original)

    def test_cli_malformed_input_produces_failure_report_not_manifest(self):
        exported, sidecar = self.root / "export.json", self.root / "sidecar.json"
        write_json(sidecar, self.sidecar)
        exported.write_text('{"not": "a full export"}', encoding="utf-8")
        output = self.root / "invalid"
        with contextlib.redirect_stdout(io.StringIO()):
            code = import_main(["--export", str(exported), "--sidecar", str(sidecar),
                                "--image-root", str(self.images), "--output", str(output)])
        self.assertEqual(code, 2)
        report = read_json(output / "review-report.json")
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertFalse((output / "manifest.json").exists())

    def test_outputs_stay_outside_served_images_and_failed_staging_is_removed(self):
        with self.assertRaises(ValueError):
            publish_annotation_files(self.images / "metadata", self.images, {"report.json": {}})
        with patch("training.annotations.write_json", side_effect=OSError("simulated storage failure")):
            with self.assertRaises(OSError):
                publish_annotation_files(self.root / "unpublished", self.images, {"report.json": {}})
        self.assertFalse((self.root / "unpublished").exists())
        self.assertEqual(list(self.root.glob("*.staging")), [])

    def test_help_does_not_import_label_studio(self):
        with patch.dict(sys.modules, {"label_studio": None}):
            for main in (prepare_main, import_main):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                    main(["--help"])
                self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()

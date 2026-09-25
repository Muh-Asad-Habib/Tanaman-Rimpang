import io
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageFile

from training.common import digest, read_json, taxonomy
from training.ingestion import (
    EXPECTED_COUNTS, EXPECTED_PROVENANCE, FOLDER_ID, METADATA_IDS,
    IngestionError, _normalize, _workspace, audit_raw_dataset, download_dataset, infer_provenance,
    parse_mapping, safe_relative,
)


def image_bytes(color, *, size=(32, 24), orientation=1, format="JPEG"):
    stream = io.BytesIO()
    image = Image.new("RGB", size, color)
    exif = Image.Exif()
    exif[274] = orientation
    image.save(stream, format=format, exif=exif)
    return stream.getvalue()


class FakeGdown:
    def __init__(self):
        self.folder_calls, self.download_calls = [], []
        self.fail_ids, self.interrupt_ids = set(), set()
        self.rows, self.payloads = [], {}
        mapping = ["\ufeff# Fixture metadata, not a training dataset", ""]
        for index, label in enumerate(taxonomy()["labels"]):
            folder = label["slug"].replace("-", " ")
            filename = label["slug"].replace("-", "_") + "_001.jpg"
            relative = f"{folder}/{filename}"
            original = f"{folder}/{folder}/IMG_{1000 + index}.JPG"
            mapping.append(f"{original} -> {relative}" if index % 2 == 0 else f"{relative} <- {original}")
            drive_id = f"fixture-image-{index:025d}"
            self.rows.append(SimpleNamespace(id=drive_id, path=relative, local_path=r"C:\untrusted\ignored.jpg"))
            self.payloads[drive_id] = image_bytes(
                (index * 20, 40, 90), orientation=6 if index == 0 else 1,
            )
        for name, drive_id in METADATA_IDS.items():
            self.rows.append(SimpleNamespace(id=drive_id, path=name, local_path=name))
            self.payloads[drive_id] = (
                "\n".join(mapping).encode("utf-8") if name == "rename_mapping.txt"
                else b"Fixture attribution only. License review remains pending.\n"
            )

    def download_folder(self, **kwargs):
        self.folder_calls.append(kwargs)
        return self.rows

    def download(self, **kwargs):
        self.download_calls.append(kwargs)
        drive_id = kwargs["id"]
        destination = Path(kwargs["output"])
        partial = destination.with_name(destination.name + ".fixture.part")
        if drive_id in self.interrupt_ids:
            self.interrupt_ids.remove(drive_id)
            partial.write_bytes(self.payloads[drive_id][:20])
            raise KeyboardInterrupt
        if drive_id in self.fail_ids:
            partial.write_bytes(self.payloads[drive_id][:20])
            raise TimeoutError("fixture read timeout")
        destination.write_bytes(self.payloads[drive_id])
        if partial.exists():
            partial.unlink()
        return str(destination)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        # Fixtures stay inside the project, never the machine's temporary directory.
        self.root = Path.cwd() / (".ingestion-test-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.data = self.root / "dataset"
        self.downloader = FakeGdown()
        self.expectations = {
            "imageCounts": {label["slug"]: 1 for label in taxonomy()["labels"]},
            "provenanceCounts": {"original": 10},
        }

    def tearDown(self):
        shutil.rmtree(self.root)

    def download(self, **kwargs):
        return download_dataset(
            self.data, expectations=self.expectations, downloader=self.downloader, **kwargs,
        )

    def manifest(self, name):
        return read_json(self.data / "manifests" / name)

    def first_image(self):
        return next(row for row in self.downloader.rows if row.path.startswith("jahe/"))

    def test_default_expectations_match_approved_snapshot(self):
        self.assertEqual(sum(EXPECTED_COUNTS.values()), 5507)
        self.assertEqual(sum(EXPECTED_PROVENANCE.values()), 5507)
        self.assertEqual(set(EXPECTED_COUNTS), {row["slug"] for row in taxonomy()["labels"]})

    def test_mapping_both_directions_comments_and_unknown_provenance(self):
        result = parse_mapping(
            "\ufeff# note\n\njahe/jahe/IMG_123.JPG -> jahe/a.jpg\n"
            "jahe merah/b.jpg <- batch2/empon_123.jpg\n"
            "kunyit/c.jpg <- unfamiliar/name.jpg\n"
        )
        self.assertEqual(result["jahe/a.jpg"]["id"], "original")
        self.assertEqual(result["jahe merah/b.jpg"]["id"], "empon")
        self.assertEqual(result["kunyit/c.jpg"]["id"], "unknown")
        self.assertTrue(all(row["licenseStatus"] == "pending" for row in result.values()))
        self.assertNotIn("groupId", result["jahe/a.jpg"])
        for invalid in (
            "", "# only comments", "not a mapping", "a -> b <- c",
            "a -> jahe/a.jpg\nb -> jahe/a.jpg", "a -> jahe/a.JPG\nb -> jahe/a.jpg",
            "a -> ../outside.jpg", r"C:\secret.jpg -> jahe/a.jpg",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(IngestionError):
                parse_mapping(invalid)

    def test_source_classification_is_pending_and_parent_is_only_a_hint(self):
        for prefix in ("dimas_v4", "empon", "spices", "rfempon", "yusuf", "upn_v2", "taufiq_v8"):
            self.assertEqual(infer_provenance(f"batch2/{prefix}_001.jpg")["id"], prefix)
        self.assertEqual(infer_provenance("uploads/jahe.jpg")["id"], "uploads")
        hint = infer_provenance("jahe/external/dimas_v4_parent_jpg.rf.abcdef_0.jpg")
        self.assertEqual(hint["parentSuggestion"]["status"], "pending")
        self.assertEqual(hint["licenseStatus"], "pending")
        self.assertEqual(infer_provenance("unknown/external/other.jpg")["id"], "unknown")

    def test_unsafe_paths_are_rejected_and_windows_separators_are_portable(self):
        self.assertEqual(safe_relative(r"jahe\a.jpg"), "jahe/a.jpg")
        for relative in (
            "../x", "a/../x", "a//x", "/x", r"\\server\x", r"C:\x", "a/./x",
            "a/CON.jpg", "a/a.jpg:stream", "a/name. ", "a/\0x", "a/*",
        ):
            with self.subTest(path=relative), self.assertRaises(IngestionError):
                safe_relative(relative)

    def test_inventory_only_checks_full_mapping_and_downloads_only_metadata(self):
        result = self.download(inventory_only=True)
        self.assertEqual(result["status"], "inventory-only")
        self.assertEqual(len(self.downloader.download_calls), 2)
        self.assertEqual({call["id"] for call in self.downloader.download_calls}, set(METADATA_IDS.values()))
        self.assertFalse((self.data / "raw").exists())
        self.assertFalse((self.data / "manifests" / "raw-inventory.json").exists())
        listing = self.downloader.folder_calls[0]
        self.assertEqual(listing["id"], FOLDER_ID)
        self.assertTrue(listing["skip_download"])
        self.assertFalse(listing["use_cookies"])
        self.assertEqual(self.manifest("reconciliation.json")["status"], "matched")

    def test_fifty_per_folder_is_not_a_complete_inventory(self):
        rows = []
        for index, label in enumerate(taxonomy()["labels"]):
            for number in range(50):
                rows.append(SimpleNamespace(
                    id=f"fixture-{index:02d}-{number:025d}",
                    path=f"{label['slug'].replace('-', ' ')}/img_{number}.jpg",
                ))
        self.downloader.rows = rows + [row for row in self.downloader.rows if row.path in METADATA_IDS]
        with self.assertRaisesRegex(IngestionError, "50-file"):
            download_dataset(self.data, downloader=self.downloader)
        self.assertEqual(self.downloader.download_calls, [])
        report = self.manifest("reconciliation.json")
        self.assertEqual(report["listedImages"], 500)
        self.assertEqual(report["expectedImages"], 5507)
        self.assertEqual(report["status"], "mismatch")

    def test_mapping_paths_and_provenance_must_both_reconcile(self):
        mapping_id = METADATA_IDS["rename_mapping.txt"]
        original = self.downloader.payloads[mapping_id]
        for replacement in (
            original.replace(b"jahe/jahe_001.jpg", b"jahe/wrong.jpg"),
            original.replace(b"jahe/jahe/IMG_1000.JPG", b"uploads/jahe.jpg"),
        ):
            self.downloader.payloads[mapping_id] = replacement
            with self.assertRaisesRegex(IngestionError, "reconcile"):
                self.download()
            self.assertFalse(any(call["id"].startswith("fixture-image") for call in self.downloader.download_calls))
            shutil.rmtree(self.data)
            self.downloader.download_calls.clear()

    def test_bad_discovery_identity_is_rejected_before_writing_outside_root(self):
        self.first_image().path = "../escape.jpg"
        with self.assertRaisesRegex(IngestionError, "Unsafe"):
            self.download()
        self.assertFalse((self.root / "escape.jpg").exists())
        self.assertEqual(self.downloader.download_calls, [])

    def test_duplicate_drive_paths_ids_and_metadata_id_changes_fail(self):
        for change in ("path", "id", "metadata"):
            with self.subTest(change=change):
                self.downloader = FakeGdown()
                if change == "path":
                    self.downloader.rows[1].path = self.downloader.rows[0].path
                elif change == "id":
                    self.downloader.rows[1].id = self.downloader.rows[0].id
                else:
                    self.downloader.rows[-1].id = "replacement-metadata-id-1234567890"
                with self.assertRaises(IngestionError):
                    self.download()
                self.assertEqual(self.downloader.download_calls, [])
                shutil.rmtree(self.data)

    def test_folder_discovery_error_surfaces_with_failed_progress(self):
        with patch.object(self.downloader, "download_folder", side_effect=RuntimeError("fixture discovery failed")):
            with self.assertRaisesRegex(RuntimeError, "discovery failed"):
                self.download()
        self.assertEqual(self.manifest("download-progress.json")["status"], "failed")
        self.assertEqual(self.downloader.download_calls, [])
        self.assertEqual(self.download()["images"], 10)

    def test_complete_transfer_is_receipted_but_does_not_promote_raw(self):
        report = self.download()
        self.assertEqual(report["status"], "downloaded-pending-audit")
        self.assertEqual(report["images"], 10)
        self.assertFalse((self.data / "raw").exists())
        inventory = self.manifest("raw-inventory.json")
        self.assertEqual(len(inventory["images"]), 10)
        for row in inventory["images"]:
            self.assertEqual(digest(self.data / "incoming" / row["path"]), row["sha256"])
            self.assertEqual(row["classProposal"]["status"], "pending")
        for call in self.downloader.download_calls:
            self.assertFalse(call["use_cookies"])
            self.assertTrue(call["resume"])
            self.assertEqual(call["timeout"], (30, 60))
            self.assertEqual(call["retries"], 3)
            self.assertNotIn("cookies_file", call)

    def test_resume_verifies_receipts_and_fetches_metadata_again(self):
        self.download()
        before = len(self.downloader.download_calls)
        self.download()
        self.assertEqual(len(self.downloader.download_calls) - before, 2)
        self.assertEqual(len(self.downloader.folder_calls), 2)
        self.assertEqual(self.manifest("download-progress.json")["completedImages"], 10)
        changed = self.first_image()
        (self.data / "incoming" / changed.path).write_bytes(b"changed outside the downloader")
        self.download()
        quarantined = self.manifest("download-progress.json")["quarantined"]
        self.assertEqual(len(quarantined), 1)
        self.assertEqual((self.data / quarantined[0]["quarantinePath"]).read_bytes(), b"changed outside the downloader")

    def test_changed_metadata_under_same_id_blocks_resume(self):
        self.download(inventory_only=True)
        self.downloader.payloads[METADATA_IDS["ATRIBUSI.txt"]] += b"Source attribution changed.\n"
        with self.assertRaisesRegex(IngestionError, "Source metadata changed"):
            self.download()
        self.assertEqual(self.manifest("download-progress.json")["status"], "failed")
        self.assertFalse((self.data / "raw").exists())

    def test_changed_ids_or_expectations_block_resume(self):
        self.download(inventory_only=True)
        self.first_image().id = "new-file-identity-12345678901234567890"
        with self.assertRaisesRegex(IngestionError, "IDs/paths"):
            self.download()

    def test_interrupted_transfer_resumes_partial_without_publishing(self):
        first = self.first_image()
        self.downloader.interrupt_ids.add(first.id)
        with self.assertRaises(KeyboardInterrupt):
            self.download()
        self.assertEqual(self.manifest("download-progress.json")["status"], "interrupted")
        self.assertTrue(list((self.data / "incoming").rglob("*.part")))
        self.assertFalse((self.data / "raw").exists())
        self.assertEqual(self.download()["images"], 10)
        self.assertFalse(list((self.data / "incoming").rglob("*.part")))
        self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")

    def test_failed_transfer_is_not_a_receipt_and_partial_audit_is_blocked(self):
        self.downloader.fail_ids.add(self.first_image().id)
        with self.assertRaisesRegex(IngestionError, "incomplete"):
            self.download()
        self.assertEqual(self.manifest("download-progress.json")["completedImages"], 9)
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["validImages"], 9)
        self.assertFalse(report["readyForAnnotation"])
        self.assertFalse((self.data / "manifests" / "normalized-inventory.json").exists())
        self.assertFalse((self.data / "raw").exists())
        self.downloader.fail_ids.clear()
        self.download()
        self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")

    def test_empty_html_and_wrong_extension_fail_transfer(self):
        first = self.first_image()
        for payload in (b"", b"<html>quota exceeded</html>", image_bytes("red", format="PNG")):
            with self.subTest(payload=payload[:20]):
                self.downloader.payloads[first.id] = payload
                with self.assertRaisesRegex(IngestionError, "incomplete"):
                    self.download()
                self.assertFalse((self.data / "raw").exists())
                self.assertEqual(self.manifest("download-progress.json")["files"][first.path]["status"], "failed")
                shutil.rmtree(self.data)

    def test_decode_exif_normalization_preserves_raw_and_pending_contract(self):
        self.download()
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["recoveredImages"], 10)
        self.assertEqual(report["validImages"], 10)
        self.assertTrue(report["readyForAnnotation"])
        self.assertFalse(report["trainingReady"])
        self.assertFalse((self.data / "incoming").exists())
        inventory = self.manifest("normalized-inventory.json")
        self.assertEqual(inventory["status"], "complete")
        self.assertEqual(inventory["rawInventorySha256"], digest(self.data / "manifests" / "raw-inventory.json"))
        first = next(row for row in inventory["images"] if row["originalPath"].startswith("jahe/"))
        self.assertEqual((first["width"], first["height"]), (24, 32))
        self.assertEqual(first["exifOrientation"], 6)
        self.assertEqual((self.data / "raw" / first["originalPath"]).read_bytes(), self.downloader.payloads[self.first_image().id])
        with Image.open(self.data / "images" / first["path"]) as image:
            self.assertEqual(image.size, (24, 32))
            self.assertEqual(image.getexif().get(274, 1), 1)
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.format, "PNG")
        for row in inventory["images"]:
            self.assertEqual(
                row["driveId"], next(item.id for item in self.downloader.rows if item.path == row["originalPath"]),
            )
            self.assertEqual(row["source"]["licenseStatus"], "pending")
            self.assertEqual(row["classProposal"]["status"], "pending")
            self.assertEqual(row["reviewStatus"], "pending")
            self.assertNotIn("objects", row)
            self.assertNotIn("groupId", row)
        self.assertEqual(self.manifest("normalized-inventory.partial.json")["status"], "partial")
        self.assertEqual(self.download()["status"], "already-promoted")

    def test_exact_duplicates_are_reported_and_retained(self):
        first, second = self.downloader.rows[:2]
        self.downloader.payloads[second.id] = self.downloader.payloads[first.id]
        self.download()
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["validImages"], 10)
        self.assertEqual(len(report["exactDuplicates"]), 1)
        self.assertTrue(report["exactDuplicates"][0]["classConflict"])
        self.assertEqual(report["exactDuplicates"][0]["status"], "pending")
        rows = self.manifest("normalized-inventory.json")["images"]
        duplicated = [row for row in rows if row["groupSuggestion"]]
        self.assertEqual(len(duplicated), 2)
        self.assertTrue(all(row["groupSuggestion"]["status"] == "pending" for row in duplicated))
        self.assertTrue((self.data / "raw" / first.path).is_file())
        self.assertTrue((self.data / "raw" / second.path).is_file())

    def test_pixel_duplicates_detect_same_image_with_different_metadata(self):
        first, second = self.downloader.rows[1:3]
        payload = self.downloader.payloads[first.id]
        # JPEG COM changes the source hash without changing any decoded pixels.
        comment = b"fixture metadata comment"
        self.downloader.payloads[second.id] = payload[:2] + b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment + payload[2:]
        self.download()
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["exactDuplicates"], [])
        self.assertEqual(len(report["pixelDuplicates"]), 1)
        self.assertEqual(set(report["pixelDuplicates"][0]["paths"]), {first.path, second.path})
        self.assertEqual(report["pixelDuplicates"][0]["status"], "pending")

    def test_near_duplicate_candidates_are_bounded_pending_suggestions(self):
        self.download()
        report = audit_raw_dataset(self.data, near_duplicates=True, max_pairs=2)
        near = report["nearDuplicates"]
        self.assertEqual(near["status"], "proposals")
        self.assertEqual(near["reviewStatus"], "pending")
        self.assertTrue(near["truncated"])
        self.assertEqual(len(near["pairs"]), 2)
        self.assertGreater(near["candidateCount"], 2)
        self.assertTrue(all(row["groupSuggestion"] is None for row in self.manifest("normalized-inventory.json")["images"]))

    def test_truncated_and_small_images_are_excluded_without_raw_promotion(self):
        first = self.first_image()
        for payload in (self.downloader.payloads[first.id][:-10], image_bytes("red", size=(15, 24))):
            with self.subTest(size=len(payload)):
                self.downloader.payloads[first.id] = payload
                self.download()
                report = audit_raw_dataset(self.data)
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual(report["recoveredImages"], 10)
                self.assertEqual(report["validImages"], 9)
                self.assertEqual(report["excludedImages"], 1)
                self.assertEqual(report["exclusions"][0]["originalPath"], first.path)
                self.assertFalse((self.data / "raw").exists())
                self.assertEqual((self.data / "incoming" / first.path).read_bytes(), payload)
                self.assertFalse((self.data / "manifests" / "normalized-inventory.json").exists())
                shutil.rmtree(self.data)

    def test_invalid_download_can_be_quarantined_and_recovered(self):
        first = self.first_image()
        good = self.downloader.payloads[first.id]
        self.downloader.payloads[first.id] = good[:-10]
        self.download()
        self.assertEqual(audit_raw_dataset(self.data)["status"], "incomplete")
        self.downloader.payloads[first.id] = good
        self.download()
        self.assertEqual(len(self.manifest("download-progress.json")["quarantined"]), 1)
        self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")

    def test_missing_or_changed_original_is_excluded_even_if_it_still_decodes(self):
        self.download()
        first = self.first_image()
        (self.data / "incoming" / first.path).write_bytes(image_bytes("green"))
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["exclusions"][0]["reason"], "checksum_mismatch")
        self.assertFalse((self.data / "raw").exists())

    def test_stray_partial_prevents_promotion_even_when_every_image_is_valid(self):
        self.download()
        (self.data / "incoming" / "leftover.part").write_bytes(b"not a complete download")
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["validImages"], 10)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("leftover.part" in error for error in report["errors"]))
        self.assertFalse((self.data / "raw").exists())

    def test_raw_duplicate_report_also_covers_decoding_failures(self):
        first, second = self.downloader.rows[:2]
        broken = self.downloader.payloads[first.id][:-10]
        self.downloader.payloads[first.id] = broken
        self.downloader.payloads[second.id] = broken
        self.download()
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["validImages"], 8)
        self.assertEqual(len(report["exactDuplicates"]), 1)
        self.assertEqual(set(report["exactDuplicates"][0]["paths"]), {first.path, second.path})

    def test_audit_interruption_resumes_verified_derivatives(self):
        self.download()
        calls = 0

        def interrupt(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return _normalize(source, destination)

        with patch("training.ingestion._normalize", side_effect=interrupt):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["validImages"], 1)
        self.assertFalse((self.data / "raw").exists())
        with patch("training.ingestion._normalize", wraps=_normalize) as normalize:
            self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")
        self.assertEqual(normalize.call_count, 9)

    def test_promotion_interruption_after_raw_rename_is_resumable(self):
        self.download()
        original_rename = Path.rename

        def fail_images(path, target):
            if path.name == "images.incoming":
                raise OSError("fixture interrupted promotion")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_images):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "failed")
        self.assertTrue((self.data / "raw").exists())
        self.assertTrue((self.data / "images.incoming").exists())
        self.assertFalse((self.data / "manifests" / "normalized-inventory.json").exists())
        self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")

    def test_reaudit_does_not_rewrite_raw_or_changed_published_derivatives(self):
        self.download()
        self.assertEqual(audit_raw_dataset(self.data)["status"], "complete")
        first = self.manifest("normalized-inventory.json")["images"][0]
        derivative = self.data / "images" / first["path"]
        derivative.write_bytes(b"tampered derivative")
        report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(derivative.read_bytes(), b"tampered derivative")
        self.assertEqual(self.manifest("normalized-inventory.json")["status"], "incomplete")
        self.assertTrue((self.data / "raw" / first["originalPath"]).is_file())

    def test_successful_reaudit_uses_verified_cache_without_rewriting_images(self):
        self.download()
        audit_raw_dataset(self.data)
        original = self.data / "raw" / self.first_image().path
        raw_mtime = original.stat().st_mtime_ns
        with patch("training.ingestion._normalize", side_effect=AssertionError("Should use verified cache")):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(original.stat().st_mtime_ns, raw_mtime)
        self.assertEqual(self.manifest("normalized-inventory.json")["status"], "complete")

    def test_unknown_files_in_raw_and_unreadable_directories_block_audit(self):
        self.download()
        walk = __import__("os").walk

        def inaccessible(top, **kwargs):
            kwargs["onerror"](PermissionError("fixture unreadable directory"))
            yield from walk(top, **kwargs)

        with patch("training.ingestion.os.walk", side_effect=inaccessible):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("Cannot inspect directory" in error for error in report["errors"]))
        self.assertFalse((self.data / "raw").exists())

    def test_workspace_lock_prevents_concurrent_download_and_audit(self):
        with _workspace(self.data):
            with self.assertRaisesRegex(IngestionError, "Another"):
                self.download()
            with self.assertRaisesRegex(IngestionError, "Another"):
                audit_raw_dataset(self.data)
        self.assertEqual(self.downloader.folder_calls, [])

    def test_lenient_pillow_mode_is_never_accepted(self):
        self.download()
        with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["validImages"], 0)
        self.assertTrue(all(row["reason"] == "unsafe_decoder" for row in report["exclusions"]))

    def test_transparent_images_require_explicit_background_review(self):
        source = self.root / "alpha.png"
        Image.new("RGBA", (20, 20), (255, 0, 0, 100)).save(source)
        with self.assertRaisesRegex(IngestionError, "background policy"):
            _normalize(source, self.root / "derivative.png")
        self.assertFalse((self.root / "derivative.png").exists())

    def test_symlink_escape_is_rejected(self):
        self.download(inventory_only=True)
        outside = self.root / "outside"
        outside.mkdir()
        link = self.data / "incoming" / "jahe"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        try:
            with self.assertRaisesRegex(IngestionError, "Symlink"):
                self.download()
            self.assertEqual(list(outside.iterdir()), [])
        finally:
            link.unlink()

    def test_symlink_guard_also_has_platform_independent_coverage(self):
        self.download()
        guarded = self.data / "incoming" / self.first_image().path
        real_is_symlink = Path.is_symlink
        with patch.object(Path, "is_symlink", autospec=True,
                          side_effect=lambda path: path == guarded or real_is_symlink(path)):
            report = audit_raw_dataset(self.data)
        self.assertEqual(report["status"], "incomplete")
        self.assertTrue(any("Symlink" in error for error in report["errors"]))
        self.assertEqual(report["validImages"], 9)
        self.assertFalse((self.data / "raw").exists())

    def test_cli_help_is_offline_and_does_not_require_gdown(self):
        for name in ("download_dataset", "audit_raw_dataset"):
            result = subprocess.run(
                [sys.executable, "-m", f"training.scripts.{name}", "--help"],
                capture_output=True, text=True, check=False, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--dataset-root", result.stdout)
        self.assertFalse(self.data.exists())

    def test_explicit_expectations_and_limits_are_validated(self):
        invalid = {"imageCounts": {"jahe": 1}, "provenanceCounts": {"original": 1}}
        with self.assertRaises(IngestionError):
            download_dataset(self.data, expectations=invalid, downloader=self.downloader)
        for kwargs in ({"retries": -1}, {"retries": 11}, {"timeout": float("nan")}, {"timeout": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(IngestionError):
                self.download(**kwargs)
        self.assertFalse(self.data.exists())


if __name__ == "__main__":
    unittest.main()

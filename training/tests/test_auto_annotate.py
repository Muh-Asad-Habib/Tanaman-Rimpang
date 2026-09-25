import unittest

from training.scripts.auto_annotate import build_groups, select_boxes


def row(path, original, dhash, parent=None, pixels=None):
    source = {"id": "dimas_v4" if parent else "original", "originalPath": original}
    if parent:
        source["parentSuggestion"] = {"key": parent}
    return {"path": path, "source": source, "dHash": dhash, "pixelSha256": pixels or path}


class AutoAnnotateTest(unittest.TestCase):
    def test_groups_merge_roboflow_parents_and_near_duplicates(self):
        rows = [
            row("a/1.png", "x/1_jpg.rf.aaa.jpg", "0000000000000000", parent="1_jpg"),
            row("a/2.png", "x/1_jpg.rf.bbb.jpg", "ffffffffffffffff", parent="1_jpg"),
            row("a/3.png", "x/3.jpg", "0000000000000003"),
            row("a/4.png", "x/4.jpg", "0f0f0f0f0f0f0f0f"),
        ]
        groups = build_groups(rows, max_distance=4)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])

    def test_select_boxes_drops_nested_parts_and_normalizes(self):
        candidates = [(0.6, [100, 100, 500, 400]), (0.5, [120, 120, 200, 200]), (0.4, [600, 100, 900, 400])]
        boxes, mode = select_boxes(candidates, 1000, 500, floor=0.2, relative=0.5, fallback=0.05)
        self.assertEqual(mode, "confident")
        self.assertEqual(len(boxes), 2)
        self.assertAlmostEqual(boxes[0][0], 0.1)
        self.assertTrue(all(x + w <= 1 and y + h <= 1 for x, y, w, h in boxes))

    def test_select_boxes_reports_missing(self):
        self.assertEqual(select_boxes([(0.01, [0, 0, 10, 10])], 100, 100, floor=0.2, relative=0.5, fallback=0.05), ([], "none"))


if __name__ == "__main__":
    unittest.main()

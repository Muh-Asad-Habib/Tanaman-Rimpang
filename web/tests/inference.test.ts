import { test } from "node:test";
import assert from "node:assert/strict";
import labels from "../../shared/labels.json";
import { fetchManifest, parseManifest } from "../src/lib/inference/manifest";
import { FrameGate } from "../src/lib/inference/scheduler";
import { ObjectTracker } from "../src/lib/inference/tracking";
import { fitPreview } from "../src/lib/preview-layout";
import { decodeDetector, intersectionOverUnion, letterboxGeometry, nonMaxSuppression, paddedCrop, softmax } from "../src/lib/inference/postprocess";

const artifact = { file: "detector.onnx", sha256: "a".repeat(64), inputName: "images", outputName: "output0", inputSize: 320, dtype: "float32" };
function readyManifest() {
  return {
    schemaVersion: 1, labelsVersion: 1, status: "ready", version: "test-only", maxObjects: 5,
    classSlugs: labels.labels.map((label) => label.slug),
    detector: { ...artifact, outputLayout: "1x5xN", padValue: 114, scoreThreshold: 0.25, iouThreshold: 0.45 },
    classifier: { ...artifact, file: "classifier.onnx", inputSize: 224, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5],
      scoreThreshold: 0.7, minMargin: 0.15, cropMargin: 0.1 },
  };
}

test("untrained manifest never becomes model-ready", () => {
  const manifest = parseManifest({ schemaVersion: 1, labelsVersion: 1, status: "unavailable", reason: "Belum dilatih." });
  assert.equal(manifest.status, "unavailable");
  assert.throws(() => parseManifest({ status: "ready" }));
});
test("ready contract validates exact labels, hashes, shapes and normalization", () => {
  assert.equal(parseManifest(readyManifest()).status, "ready");
  for (const mutate of [
    (v: ReturnType<typeof readyManifest>) => { v.classSlugs.reverse(); },
    (v: ReturnType<typeof readyManifest>) => { v.detector.file = "../detector.onnx"; },
    (v: ReturnType<typeof readyManifest>) => { v.classifier.std[0] = 0; },
    (v: ReturnType<typeof readyManifest>) => { v.detector.inputSize = 32.5; },
    (v: ReturnType<typeof readyManifest>) => { v.detector.sha256 = "fake"; },
    (v: ReturnType<typeof readyManifest>) => { v.classifier.file = v.detector.file; },
  ]) {
    const candidate = readyManifest(); mutate(candidate);
    assert.throws(() => parseManifest(candidate));
  }
});
test("availability and inference read the same uncached, validated manifest", async (context) => {
  let payload: unknown = { schemaVersion: 1, labelsVersion: 1, status: "unavailable", reason: "Bundle belum dipasang." };
  const request = context.mock.method(globalThis, "fetch", async () => Response.json(payload));
  const controller = new AbortController();
  const unavailable = await fetchManifest(controller.signal);
  assert.equal(unavailable.status, "unavailable");
  assert.equal(request.mock.calls[0].arguments[0], "/models/manifest.json");
  assert.deepEqual(request.mock.calls[0].arguments[1], { cache: "no-store", signal: controller.signal });

  payload = readyManifest();
  assert.equal((await fetchManifest()).status, "ready");
  assert.equal(request.mock.callCount(), 2);

  payload = { ...readyManifest(), classSlugs: ["wrong-order"] };
  await assert.rejects(fetchManifest(), /Kontrak model tidak valid/);
});
test("manifest fetch failures are not reported as unavailable or ready", async (context) => {
  const request = context.mock.method(globalThis, "fetch", async () => new Response("", { status: 503 }));
  await assert.rejects(fetchManifest(), /Manifest model tidak dapat dibuka \(503\)/);
  request.mock.mockImplementation(async () => new Response("not json"));
  await assert.rejects(fetchManifest(), SyntaxError);
  request.mock.mockImplementation(async () => { throw new TypeError("Network unavailable"); });
  await assert.rejects(fetchManifest(), /Network unavailable/);
});
test("manifest cancellation remains available to unmount and retry cleanup", async (context) => {
  context.mock.method(globalThis, "fetch", async (_input: string | URL | Request, init?: RequestInit) => {
    if (init?.signal?.aborted) throw new DOMException("Aborted", "AbortError");
    return Response.json(readyManifest());
  });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(fetchManifest(controller.signal), { name: "AbortError" });
});
test("letterbox and inverse coordinates preserve original frame aspect", () => {
  const geometry = letterboxGeometry(640, 360, 320);
  assert.equal(geometry.top, 70);
  const values = new Float32Array([160, 160, 320, 180, 0.9]);
  const [detection] = decodeDetector(values, [1, 5, 1], geometry, 0.2, 0.5);
  assert.deepEqual(detection.box, { x: 0, y: 0, width: 1, height: 1 });
  const portrait = letterboxGeometry(360, 640, 320);
  assert.equal(portrait.left, 70);
  assert.throws(() => letterboxGeometry(0, 1, 320));
  assert.throws(() => decodeDetector(values, [1, 6, 1], geometry, 0.2, 0.5));
});
test("preview preserves image and overlay geometry through screen and camera rotation", () => {
  for (const [stageWidth, stageHeight] of [[286, 230], [354, 265.5], [718, 430], [730, 336], [640, 480], [975, 522], [820, 180]]) {
    for (const frameRatio of [4 / 3, 3 / 4, 16 / 9, 9 / 16, 1]) {
      const fit = fitPreview(frameRatio, stageWidth / stageHeight);
      const width = parseFloat(fit.width) / 100 * stageWidth;
      const height = parseFloat(fit.height) / 100 * stageHeight;
      assert.ok(Math.abs(width / height - frameRatio) < 1e-10);
      assert.ok(width <= stageWidth + 1e-10 && height <= stageHeight + 1e-10);
      assert.ok(Math.abs(width - stageWidth) < 1e-10 || Math.abs(height - stageHeight) < 1e-10);
      const box = { x: 0.1, y: 0.2, width: 0.5, height: 0.6 };
      const offsetX = (stageWidth - width) / 2;
      const offsetY = (stageHeight - height) / 2;
      assert.ok(offsetX + (box.x + box.width) * width <= stageWidth);
      assert.ok(offsetY + (box.y + box.height) * height <= stageHeight);
    }
  }
  assert.deepEqual(fitPreview(1, 1), { width: "100%", height: "100%" });
  for (const invalid of [0, -1, NaN, Infinity]) {
    assert.throws(() => fitPreview(invalid, 1));
    assert.throws(() => fitPreview(1, invalid));
  }
});
test("NMS keeps separate objects and clamps crop padding", () => {
  const a = { box: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 }, score: 0.8 };
  const b = { box: { x: 0.6, y: 0.6, width: 0.2, height: 0.2 }, score: 0.7 };
  assert.equal(nonMaxSuppression([a, a, b], 0.5).length, 2);
  assert.equal(intersectionOverUnion(a.box, b.box), 0);
  assert.deepEqual(paddedCrop({ x: 0, y: 0, width: 1, height: 1 }, 640, 360, 0.1),
    { x: 0, y: 0, width: 640, height: 360 });
});
test("frame backpressure and resets cannot unlock another session's work", () => {
  const gate = new FrameGate();
  const first = gate.begin()!;
  assert.equal(gate.begin(), null);
  gate.reset();
  const second = gate.begin()!;
  assert.equal(gate.finish(first.sessionId, first.frameId), false);
  assert.equal(gate.begin(), null);
  assert.equal(gate.finish(second.sessionId, second.frameId), true);
  assert.notEqual(gate.begin(), null);
});
test("per-object labels never share smoothing and reset on disappearance", () => {
  const tracker = new ObjectTracker();
  const detections = [
    { box: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 }, score: 0.9 },
    { box: { x: 0.6, y: 0.6, width: 0.2, height: 0.2 }, score: 0.8 },
  ];
  const scores = (id: number) => Array.from({ length: 10 }, (_, i) => i === id ? 0.91 : 0.01);
  const initial = tracker.update(detections, 0, 5);
  assert.equal(tracker.classify(initial[0].id, scores(0), 0.7, 0.1).status, "analyzing");
  tracker.classify(initial[1].id, scores(3), 0.7, 0.1);
  const second = tracker.update(detections, 250, 5);
  assert.equal(tracker.classify(second[0].id, scores(0), 0.7, 0.1).classId, 0);
  assert.equal(tracker.classify(second[1].id, scores(3), 0.7, 0.1).classId, 3);
  tracker.update([], 500, 5);
  const fresh = tracker.update(detections, 750, 5);
  assert.equal(tracker.classify(fresh[0].id, scores(0), 0.7, 0.1).status, "analyzing");
});
test("softmax is finite for large logits and rejects wrong output", () => {
  const result = softmax(new Float32Array([1000, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  assert.equal(result[0], 1);
  assert.throws(() => softmax(new Float32Array([1, 2])));
});

import * as ort from "onnxruntime-web/webgpu";
import type { ReadyManifest, WorkerRequest, WorkerResponse, Provider, Artifact } from "./contracts";
import { detectorInput, classifierInput } from "./preprocess";
import { decodeDetector, softmax } from "./postprocess";
import { ObjectTracker } from "./tracking";

let detector: ort.InferenceSession | null = null;
let classifier: ort.InferenceSession | null = null;
let manifest: ReadyManifest | null = null;
let session = -1;
let busy = false;
let pending: Extract<WorkerRequest, { type: "frame" }> | null = null;
const tracker = new ObjectTracker();
const send = (response: WorkerResponse) => self.postMessage(response);

async function loadArtifact(baseUrl: string, asset: Artifact): Promise<ArrayBuffer> {
  const response = await fetch(new URL(asset.file, baseUrl));
  if (!response.ok) throw new Error(`Artefak ${asset.file} tidak dapat dibuka (${response.status}).`);
  const data = await response.arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", data)))
    .map((byte) => byte.toString(16).padStart(2, "0")).join("");
  if (hash !== asset.sha256) throw new Error(`Checksum ${asset.file} tidak sesuai manifest.`);
  return data;
}

async function initialize(config: ReadyManifest, baseUrl: string): Promise<void> {
  manifest = config;
  ort.env.wasm.wasmPaths = new URL("/vendor/ort/", baseUrl).href;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  const detectorBytes = await loadArtifact(baseUrl, config.detector);
  const classifierBytes = await loadArtifact(baseUrl, config.classifier);
  let provider: Provider = "wasm";
  let note: string | undefined;
  async function createSessions(ep: Provider) {
    detector = await ort.InferenceSession.create(detectorBytes, { executionProviders: [ep] });
    try {
      classifier = await ort.InferenceSession.create(classifierBytes, { executionProviders: [ep] });
    } catch (error) {
      await detector.release();
      detector = null;
      throw error;
    }
  }
  if ("gpu" in navigator) {
    try {
      await createSessions("webgpu");
      provider = "webgpu";
    } catch (error) {
      note = `WebGPU tidak tersedia untuk model ini. Menggunakan CPU/WASM; kecepatan dapat berkurang. ${error instanceof Error ? error.message : "Inisialisasi GPU gagal."}`;
      await createSessions("wasm");
    }
  } else {
    note = "Model siap melalui CPU/WASM. Perangkat ini belum tentu memenuhi target realtime.";
    await createSessions("wasm");
  }
  send({ type: "ready", provider, note });
}

async function infer(message: Extract<WorkerRequest, { type: "frame" }>) {
  if (!manifest || !detector || !classifier) throw new Error("Model belum diinisialisasi.");
  if (session === -1) {
    session = message.sessionId;
    tracker.reset();
  }
  if (message.sessionId !== session) return;
  const start = performance.now();
  const preprocessed = detectorInput(message.bitmap, manifest.detector);
  const detectorTensor = new ort.Tensor("float32", preprocessed.data, [1, 3, manifest.detector.inputSize, manifest.detector.inputSize]);
  let detected: ReturnType<typeof decodeDetector>;
  try {
    const outputs = await detector.run({ [manifest.detector.inputName]: detectorTensor });
    try {
      const output = outputs[manifest.detector.outputName];
      if (!output || !(output.data instanceof Float32Array)) throw new Error("Tensor detector tidak cocok dengan manifest.");
      detected = decodeDetector(output.data, output.dims, preprocessed.geometry, manifest.detector.scoreThreshold, manifest.detector.iouThreshold);
    } finally { Object.values(outputs).forEach((tensor) => tensor.dispose()); }
  } finally { detectorTensor.dispose(); }
  if (session !== message.sessionId) return;
  const tracks = tracker.update(detected, message.timestamp, manifest.maxObjects);
  const objects = [];
  // Every selected object is classified once per cycle, not just the highest-scoring box.
  for (const track of tracks) {
    if (session !== message.sessionId) return;
    const tensor = new ort.Tensor("float32", classifierInput(message.bitmap, track.detection.box, manifest.classifier),
      [1, 3, manifest.classifier.inputSize, manifest.classifier.inputSize]);
    try {
      const outputs = await classifier.run({ [manifest.classifier.inputName]: tensor });
      try {
        if (session !== message.sessionId) return;
        const output = outputs[manifest.classifier.outputName];
        if (!output || output.dims.length !== 2 || output.dims[0] !== 1 || output.dims[1] !== 10
          || !(output.data instanceof Float32Array)) throw new Error("Tensor classifier tidak cocok dengan kontrak 1x10.");
        objects.push(tracker.classify(track.id, softmax(output.data), manifest.classifier.scoreThreshold, manifest.classifier.minMargin));
      } finally { Object.values(outputs).forEach((output) => output.dispose()); }
    } finally { tensor.dispose(); }
  }
  if (session === message.sessionId) send({ type: "result", result: {
    sessionId: session, frameId: message.frameId, timestamp: message.timestamp, objects,
    overflow: detected.length > manifest.maxObjects, inferenceMs: performance.now() - start,
  } });
}

async function processFrame(message: Extract<WorkerRequest, { type: "frame" }>): Promise<void> {
  busy = true;
  try {
    await infer(message);
  } catch (error) {
    if (session === message.sessionId) {
      send({ type: "error", message: error instanceof Error ? error.message : "Inferensi gagal." });
    }
  } finally {
    message.bitmap.close();
    busy = false;
    const next = pending;
    pending = null;
    if (next) void processFrame(next);
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type === "reset") {
    session = message.sessionId;
    tracker.reset();
    pending?.bitmap.close();
    pending = null;
  } else if (message.type === "initialize") {
    void initialize(message.manifest, message.baseUrl).catch((error: unknown) => {
      send({ type: "error", message: error instanceof Error ? error.message : "Inisialisasi model gagal." });
    });
  } else if (busy) {
    // A new session may arrive before an old GPU call has finished. Keep only the newest frame.
    pending?.bitmap.close();
    pending = message;
  } else {
    void processFrame(message);
  }
};

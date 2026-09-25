import taxonomy from "../../../../shared/labels.json";
import type { Artifact, Manifest } from "./contracts";

export async function fetchManifest(signal?: AbortSignal): Promise<Manifest> {
  const response = await fetch("/models/manifest.json", { cache: "no-store", signal });
  if (!response.ok) throw new Error(`Manifest model tidak dapat dibuka (${response.status}).`);
  return parseManifest(await response.json());
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function finite(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max;
}
function artifact(value: unknown): value is Artifact & Record<string, unknown> {
  return record(value)
    && typeof value.file === "string" && /^[a-zA-Z0-9_.-]+\.onnx$/.test(value.file)
    && typeof value.sha256 === "string" && /^[a-f0-9]{64}$/.test(value.sha256)
    && typeof value.inputName === "string" && value.inputName.length > 0
    && typeof value.outputName === "string" && value.outputName.length > 0
    && finite(value.inputSize, 32, 1024) && Number.isInteger(value.inputSize)
    && value.dtype === "float32";
}
function triple(value: unknown, positive = false): value is [number, number, number] {
  return Array.isArray(value) && value.length === 3
    && value.every((v) => typeof v === "number" && Number.isFinite(v) && (!positive || v > 0));
}

export function parseManifest(value: unknown): Manifest {
  if (!record(value) || value.schemaVersion !== 1 || value.labelsVersion !== taxonomy.version) {
    throw new Error("Versi manifest model tidak cocok dengan aplikasi.");
  }
  if (value.status === "unavailable" && typeof value.reason === "string" && value.reason.trim()) {
    return { schemaVersion: 1, labelsVersion: 1, status: "unavailable", reason: value.reason };
  }
  const { detector, classifier } = value;
  const expected = taxonomy.labels.map((label) => label.slug);
  if (value.status !== "ready" || typeof value.version !== "string" || !value.version.trim()
    || value.maxObjects !== 5 || !Array.isArray(value.classSlugs)
    || JSON.stringify(value.classSlugs) !== JSON.stringify(expected)
    || !artifact(detector) || !artifact(classifier)
    || detector.outputLayout !== "1x5xN" || detector.padValue !== 114
    || !finite(detector.scoreThreshold, 0, 1) || !finite(detector.iouThreshold, Number.EPSILON, 1)
    || !triple(classifier.mean) || !triple(classifier.std, true)
    || !finite(classifier.cropMargin, 0, 0.5)
    || !finite(classifier.scoreThreshold, 0, 1) || !finite(classifier.minMargin, 0, 1)) {
    throw new Error("Kontrak model tidak valid. Periksa kelas, tensor, normalisasi, dan ambang model.");
  }
  if (detector.file === classifier.file) throw new Error("Detector dan classifier harus memakai artefak berbeda.");
  return {
    schemaVersion: 1, labelsVersion: 1, status: "ready",
    version: value.version, classSlugs: expected, maxObjects: 5,
    ...(value.experimental === true && typeof value.reason === "string" && value.reason.trim()
      ? { experimentalNote: value.reason } : {}),
    detector: { ...detector, outputLayout: "1x5xN", padValue: 114,
      scoreThreshold: detector.scoreThreshold, iouThreshold: detector.iouThreshold },
    classifier: { ...classifier, mean: classifier.mean, std: classifier.std,
      cropMargin: classifier.cropMargin, scoreThreshold: classifier.scoreThreshold, minMargin: classifier.minMargin },
  };
}

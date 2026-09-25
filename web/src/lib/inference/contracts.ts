export type EngineStatus = "unavailable" | "loading" | "ready" | "error";
export type Provider = "webgpu" | "wasm";
export type Box = { x: number; y: number; width: number; height: number };

export interface Artifact {
  file: string;
  sha256: string;
  inputName: string;
  outputName: string;
  inputSize: number;
  dtype: "float32";
}
export interface ReadyManifest {
  schemaVersion: 1;
  labelsVersion: 1;
  status: "ready";
  version: string;
  experimentalNote?: string;
  classSlugs: string[];
  maxObjects: 5;
  detector: Artifact & {
    outputLayout: "1x5xN";
    scoreThreshold: number;
    iouThreshold: number;
    padValue: 114;
  };
  classifier: Artifact & {
    mean: [number, number, number];
    std: [number, number, number];
    cropMargin: number;
    scoreThreshold: number;
    minMargin: number;
  };
}
export type Manifest = ReadyManifest | {
  schemaVersion: 1;
  labelsVersion: 1;
  status: "unavailable";
  reason: string;
};
export interface IdentifiedObject {
  trackId: number;
  /** Coordinates normalized to the original, unmirrored frame. */
  box: Box;
  classId: number | null;
  confidence: number | null;
  detectorConfidence: number;
  status: "analyzing" | "recognized" | "uncertain";
  source: "model";
}
export interface FrameResult {
  sessionId: number;
  frameId: number;
  timestamp: number;
  objects: IdentifiedObject[];
  overflow: boolean;
  inferenceMs: number;
}
export type WorkerRequest =
  | { type: "initialize"; manifest: ReadyManifest; baseUrl: string }
  | { type: "frame"; bitmap: ImageBitmap; sessionId: number; frameId: number; timestamp: number }
  | { type: "reset"; sessionId: number };
export type WorkerResponse =
  | { type: "ready"; provider: Provider; note?: string }
  | { type: "result"; result: FrameResult }
  | { type: "error"; message: string };

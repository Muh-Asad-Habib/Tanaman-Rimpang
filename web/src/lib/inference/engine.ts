import { fetchManifest } from "./manifest";
import { FrameGate } from "./scheduler";
import type { FrameResult, Provider, WorkerRequest, WorkerResponse } from "./contracts";

export interface EngineState {
  status: "unavailable" | "loading" | "ready" | "error";
  message: string;
  provider?: Provider;
}

export class InferenceEngine {
  private worker: Worker | null = null;
  private gate = new FrameGate();
  private state: EngineState["status"] = "unavailable";
  private generation = 0;
  private disposed = false;
  private controller: AbortController | null = null;
  private timeout: ReturnType<typeof setTimeout> | null = null;

  constructor(private onResult: (result: FrameResult) => void, private onState: (state: EngineState) => void) {}

  private report(state: EngineState) {
    this.state = state.status;
    if (!this.disposed) this.onState(state);
  }

  async initialize(): Promise<void> {
    if (this.disposed) return;
    const generation = ++this.generation;
    this.controller?.abort();
    this.worker?.terminate();
    this.worker = null;
    this.clearTimeout();
    this.gate.reset();
    this.controller = new AbortController();
    this.report({ status: "loading", message: "Memeriksa ketersediaan model..." });
    try {
      const manifest = await fetchManifest(this.controller.signal);
      if (generation !== this.generation || this.disposed) return;
      if (manifest.status === "unavailable") {
        this.report({ status: "unavailable", message: manifest.reason });
        return;
      }
      if (!window.isSecureContext || typeof Worker === "undefined" || typeof OffscreenCanvas === "undefined") {
        throw new Error("Inferensi memerlukan HTTPS dan browser yang mendukung Worker serta OffscreenCanvas.");
      }
      const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      this.worker = worker;
      worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        if (generation !== this.generation || this.disposed) return;
        const message = event.data;
        if (message.type === "ready") {
          this.clearTimeout();
          const base = message.note ?? `Model siap, diproses lokal melalui ${message.provider.toUpperCase()}.`;
          this.report({ status: "ready", provider: message.provider,
            message: manifest.experimentalNote ? `${base} ${manifest.experimentalNote}` : base });
        } else if (message.type === "result") {
          if (this.gate.finish(message.result.sessionId, message.result.frameId)) {
            this.clearTimeout();
            this.onResult(message.result);
          }
        } else {
          this.fail(message.message);
        }
      };
      worker.onerror = (event) => this.fail(event.message || "Worker inferensi berhenti. Muat ulang model.");
      worker.postMessage({ type: "initialize", manifest, baseUrl: new URL("/models/", window.location.origin).href } satisfies WorkerRequest);
      this.timeout = setTimeout(() => this.fail("Pemuatan model terlalu lama. Periksa koneksi lalu coba kembali."), 120000);
    } catch (error) {
      if (generation !== this.generation || this.disposed || this.controller?.signal.aborted) return;
      this.fail(error instanceof Error ? error.message : "Model tidak dapat dimuat.");
    }
  }

  async processFrame(source: HTMLVideoElement | HTMLImageElement): Promise<void> {
    if (this.disposed || this.state !== "ready" || !this.worker) return;
    const token = this.gate.begin();
    if (!token) return;
    try {
      const bitmap = await createImageBitmap(source);
      if (!this.gate.isCurrent(token.sessionId) || !this.worker || this.disposed) {
        bitmap.close();
        return;
      }
      this.worker.postMessage({ type: "frame", ...token, timestamp: performance.now(), bitmap } satisfies WorkerRequest, [bitmap]);
      this.timeout = setTimeout(() => this.fail("Inferensi tidak merespons. Hentikan kamera dan muat ulang model."), 30000);
    } catch (error) {
      if (this.gate.finish(token.sessionId, token.frameId)) {
        this.fail(error instanceof Error ? error.message : "Frame kamera tidak dapat dibaca.");
      }
    }
  }

  reset(): void {
    if (this.state !== "loading") this.clearTimeout();
    const sessionId = this.gate.reset();
    this.worker?.postMessage({ type: "reset", sessionId } satisfies WorkerRequest);
  }

  private clearTimeout(): void {
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = null;
  }
  private fail(message: string): void {
    this.clearTimeout();
    this.gate.reset();
    this.worker?.terminate();
    this.worker = null;
    this.report({ status: "error", message });
  }
  dispose(): void {
    this.disposed = true;
    this.generation++;
    this.controller?.abort();
    this.clearTimeout();
    this.gate.reset();
    this.worker?.terminate();
    this.worker = null;
  }
}

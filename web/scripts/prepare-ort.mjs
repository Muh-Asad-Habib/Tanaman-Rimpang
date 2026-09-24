import { copyFile, mkdir } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destination = path.join(root, "public", "vendor", "ort");
await mkdir(destination, { recursive: true });
for (const name of [
  "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.jsep.wasm", "ort-wasm-simd-threaded.jsep.mjs",
]) {
  await copyFile(require.resolve(`onnxruntime-web/${name}`), path.join(destination, name));
}
console.log("Aset ONNX Runtime lokal siap (WASM dan WebGPU, satu versi).");

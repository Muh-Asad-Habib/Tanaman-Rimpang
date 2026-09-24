import type { Box } from "./contracts";

export interface Detection { box: Box; score: number }
export interface Letterbox {
  originalWidth: number;
  originalHeight: number;
  resizedWidth: number;
  resizedHeight: number;
  left: number;
  top: number;
}

export function letterboxGeometry(width: number, height: number, size: number): Letterbox {
  if (![width, height, size].every((v) => Number.isFinite(v) && v > 0)) {
    throw new Error("Dimensi gambar harus positif.");
  }
  const ratio = Math.min(size / width, size / height);
  const resizedWidth = Math.max(1, Math.round(width * ratio));
  const resizedHeight = Math.max(1, Math.round(height * ratio));
  return { originalWidth: width, originalHeight: height, resizedWidth, resizedHeight,
    left: Math.floor((size - resizedWidth) / 2), top: Math.floor((size - resizedHeight) / 2) };
}

export function intersectionOverUnion(a: Box, b: Box): number {
  const intersection = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x))
    * Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
  const union = a.width * a.height + b.width * b.height - intersection;
  return union > 0 ? intersection / union : 0;
}

export function nonMaxSuppression(boxes: Detection[], threshold: number, limit = 30): Detection[] {
  const pending = [...boxes].sort((a, b) => b.score - a.score).slice(0, 300);
  const selected: Detection[] = [];
  for (const candidate of pending) {
    if (!selected.some((box) => intersectionOverUnion(box.box, candidate.box) > threshold)) {
      selected.push(candidate);
      if (selected.length >= limit) break;
    }
  }
  return selected;
}

export function decodeDetector(
  values: Float32Array, dims: readonly number[], geometry: Letterbox, scoreThreshold: number, iouThreshold: number,
): Detection[] {
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== 5 || values.length !== dims[2] * 5) {
    throw new Error("Output detector bukan tensor 1x5xN. Ekspor ulang model generik tanpa NMS.");
  }
  const count = dims[2];
  const boxes: Detection[] = [];
  for (let i = 0; i < count; i++) {
    const score = values[4 * count + i];
    const cx = values[i], cy = values[count + i], width = values[2 * count + i], height = values[3 * count + i];
    if (![score, cx, cy, width, height].every(Number.isFinite)) {
      throw new Error("Detector menghasilkan nilai tidak finite.");
    }
    if (score < scoreThreshold || score > 1 || width <= 0 || height <= 0) continue;
    const x1 = Math.max(0, Math.min(1, (cx - width / 2 - geometry.left) / geometry.resizedWidth));
    const y1 = Math.max(0, Math.min(1, (cy - height / 2 - geometry.top) / geometry.resizedHeight));
    const x2 = Math.max(0, Math.min(1, (cx + width / 2 - geometry.left) / geometry.resizedWidth));
    const y2 = Math.max(0, Math.min(1, (cy + height / 2 - geometry.top) / geometry.resizedHeight));
    if (x2 > x1 && y2 > y1) boxes.push({ box: { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }, score });
  }
  return nonMaxSuppression(boxes, iouThreshold);
}

export function softmax(logits: Float32Array): number[] {
  if (logits.length !== 10 || !logits.every(Number.isFinite)) {
    throw new Error("Classifier harus menghasilkan 10 logits finite.");
  }
  const maximum = Math.max(...logits);
  const scores = Array.from(logits, (logit) => Math.exp(logit - maximum));
  const total = scores.reduce((sum, score) => sum + score, 0);
  return scores.map((score) => score / total);
}

export function paddedCrop(box: Box, width: number, height: number, margin: number): Box {
  const x = Math.max(0, Math.floor((box.x - box.width * margin) * width));
  const y = Math.max(0, Math.floor((box.y - box.height * margin) * height));
  const right = Math.min(width, Math.ceil((box.x + box.width * (1 + margin)) * width));
  const bottom = Math.min(height, Math.ceil((box.y + box.height * (1 + margin)) * height));
  if (right <= x || bottom <= y) throw new Error("Crop objek kosong.");
  return { x, y, width: right - x, height: bottom - y };
}

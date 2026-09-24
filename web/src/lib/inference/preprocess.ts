import type { Box, ReadyManifest } from "./contracts";
import { letterboxGeometry, paddedCrop } from "./postprocess";

function tensorData(image: ImageData, mean: readonly number[], std: readonly number[]): Float32Array {
  const pixels = image.width * image.height;
  const data = new Float32Array(pixels * 3);
  for (let p = 0; p < pixels; p++) {
    for (let channel = 0; channel < 3; channel++) {
      data[channel * pixels + p] = (image.data[p * 4 + channel] / 255 - mean[channel]) / std[channel];
    }
  }
  return data;
}

function context(size: number): OffscreenCanvasRenderingContext2D {
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Pemrosesan gambar OffscreenCanvas tidak tersedia.");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  return ctx;
}

export function detectorInput(bitmap: ImageBitmap, config: ReadyManifest["detector"]) {
  const { inputSize } = config;
  const geometry = letterboxGeometry(bitmap.width, bitmap.height, inputSize);
  const ctx = context(inputSize);
  ctx.fillStyle = `rgb(${config.padValue} ${config.padValue} ${config.padValue})`;
  ctx.fillRect(0, 0, inputSize, inputSize);
  ctx.drawImage(bitmap, geometry.left, geometry.top, geometry.resizedWidth, geometry.resizedHeight);
  return { data: tensorData(ctx.getImageData(0, 0, inputSize, inputSize), [0, 0, 0], [1, 1, 1]), geometry };
}

export function classifierInput(bitmap: ImageBitmap, box: Box, config: ReadyManifest["classifier"]) {
  const crop = paddedCrop(box, bitmap.width, bitmap.height, config.cropMargin);
  const ctx = context(config.inputSize);
  ctx.drawImage(bitmap, crop.x, crop.y, crop.width, crop.height, 0, 0, config.inputSize, config.inputSize);
  return tensorData(ctx.getImageData(0, 0, config.inputSize, config.inputSize), config.mean, config.std);
}

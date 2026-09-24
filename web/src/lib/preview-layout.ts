export function fitPreview(frameRatio: number, containerRatio: number) {
  if (![frameRatio, containerRatio].every((ratio) => Number.isFinite(ratio) && ratio > 0)) {
    throw new Error("Rasio pratinjau harus bernilai positif dan terbatas.");
  }
  return frameRatio >= containerRatio
    ? { width: "100%", height: `${containerRatio / frameRatio * 100}%` }
    : { width: `${frameRatio / containerRatio * 100}%`, height: "100%" };
}

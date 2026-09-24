import type { Detection } from "./postprocess";
import { intersectionOverUnion } from "./postprocess";
import type { IdentifiedObject } from "./contracts";

interface Track {
  id: number;
  detection: Detection;
  seenAt: number;
  scores: number[] | null;
  candidate: number | null;
  confirmations: number;
}

export class ObjectTracker {
  private tracks: Track[] = [];
  private nextId = 1;

  reset(): void { this.tracks = []; this.nextId = 1; }

  update(detections: Detection[], timestamp: number, maxObjects: number): Track[] {
    const prior = this.tracks.filter((track) => timestamp - track.seenAt <= 1000);
    const used = new Set<number>();
    const ranked = [...detections].sort((a, b) => {
      const continuityA = prior.some((p) => intersectionOverUnion(p.detection.box, a.box) > 0.4) ? 1 : 0;
      const continuityB = prior.some((p) => intersectionOverUnion(p.detection.box, b.box) > 0.4) ? 1 : 0;
      return continuityB - continuityA || b.score - a.score;
    }).slice(0, maxObjects);
    this.tracks = ranked.map((detection) => {
      const candidates = prior.filter((p) => !used.has(p.id))
        .map((track) => ({ track, iou: intersectionOverUnion(track.detection.box, detection.box) }))
        .filter((item) => item.iou >= 0.4).sort((a, b) => b.iou - a.iou);
      const best = candidates[0];
      const ambiguous = best && (
        (candidates[1] && best.iou - candidates[1].iou < 0.15)
        || ranked.some((other) => other !== detection && intersectionOverUnion(other.box, best.track.detection.box) >= best.iou - 0.15)
      );
      if (best && !ambiguous) {
        used.add(best.track.id);
        return { ...best.track, detection, seenAt: timestamp };
      }
      return { id: this.nextId++, detection, seenAt: timestamp, scores: null, candidate: null, confirmations: 0 };
    });
    return this.tracks;
  }

  classify(id: number, scores: number[], threshold: number, minMargin: number): IdentifiedObject {
    const track = this.tracks.find((item) => item.id === id);
    if (!track) throw new Error("Track klasifikasi sudah tidak aktif.");
    track.scores = track.scores ? scores.map((p, i) => 0.65 * p + 0.35 * track.scores![i]) : [...scores];
    const ranking = track.scores.map((score, classId) => ({ score, classId })).sort((a, b) => b.score - a.score);
    const top = ranking[0];
    const accepted = top.score >= threshold && top.score - ranking[1].score >= minMargin;
    track.confirmations = accepted && top.classId === track.candidate ? track.confirmations + 1 : accepted ? 1 : 0;
    track.candidate = accepted ? top.classId : null;
    const recognized = track.confirmations >= 2;
    return { trackId: id, box: track.detection.box, detectorConfidence: track.detection.score,
      classId: recognized ? top.classId : null, confidence: recognized ? top.score : null,
      status: recognized ? "recognized" : accepted ? "analyzing" : "uncertain", source: "model" };
  }
}

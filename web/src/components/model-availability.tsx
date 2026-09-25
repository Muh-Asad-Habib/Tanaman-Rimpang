"use client";

import { useEffect, useState } from "react";
import { fetchManifest } from "@/lib/inference/manifest";
import { Icon } from "./icons";

type Availability = {
  status: "loading" | "ready" | "unavailable" | "error";
  message: string;
};

export function ModelAvailability() {
  const [attempt, setAttempt] = useState(0);
  const [availability, setAvailability] = useState<Availability>({
    status: "loading",
    message: "Memeriksa manifest model terbaru.",
  });

  useEffect(() => {
    const controller = new AbortController();
    void fetchManifest(controller.signal).then((manifest) => {
      if (controller.signal.aborted) return;
      setAvailability({
        status: manifest.status,
        message: manifest.status === "ready"
          ? "Bundle tersedia. Buka ruang pindai untuk memuat model dan memeriksa dukungan perangkat."
          : manifest.reason,
      });
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setAvailability({
        status: "error",
        message: error instanceof Error ? error.message : "Manifest model tidak dapat diperiksa.",
      });
    });
    return () => controller.abort();
  }, [attempt]);

  const title = availability.status === "ready" ? "Bundle model tersedia"
    : availability.status === "unavailable" ? "Saat ini: pratinjau saja"
    : availability.status === "error" ? "Status model belum diketahui"
    : "Memeriksa status model";

  return <div className="model-availability" data-status={availability.status}>
    <div role="status" aria-live="polite">
      <h3><Icon name={availability.status === "ready" ? "check" : "info"} size={18} />{title}</h3>
      <p>{availability.message}</p>
    </div>
    <button className="retry-button" disabled={availability.status === "loading"} onClick={() => {
      setAvailability({ status: "loading", message: "Memeriksa manifest model terbaru." });
      setAttempt((value) => value + 1);
    }}>Periksa kembali</button>
  </div>;
}

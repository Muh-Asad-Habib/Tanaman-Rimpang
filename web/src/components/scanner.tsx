"use client";

import Image from "next/image";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { InferenceEngine, type EngineState } from "@/lib/inference/engine";
import type { FrameResult } from "@/lib/inference/contracts";
import { plants } from "@/lib/catalog";
import { fitPreview } from "@/lib/preview-layout";
import { Icon } from "./icons";

type Mode = "camera" | "photo" | "demo";
const demoObjects = [
  { plant: plants[0], box: { x: 0.06, y: 0.16, width: 0.39, height: 0.48 } },
  { plant: plants[3], box: { x: 0.51, y: 0.21, width: 0.42, height: 0.29 } },
  { plant: plants[2], box: { x: 0.52, y: 0.60, width: 0.24, height: 0.26 } },
];
const position = (box: { x: number; y: number; width: number; height: number }) =>
  ({ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` });

export function Scanner() {
  const [mode, setMode] = useState<Mode>("camera");
  const [active, setActive] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [error, setError] = useState("");
  const [photo, setPhoto] = useState<string | null>(null);
  const [ratio, setRatio] = useState(4 / 3);
  const [stageRatio, setStageRatio] = useState(4 / 3);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [state, setState] = useState<EngineState>({ status: "loading", message: "Memeriksa ketersediaan model..." });
  const [result, setResult] = useState<FrameResult | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const objectUrl = useRef<string | null>(null);
  const engine = useRef<InferenceEngine | null>(null);
  const operation = useRef(0);
  const mounted = useRef(true);
  const modeRef = useRef<Mode>("camera");

  const stop = useCallback(() => {
    operation.current++;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    engine.current?.reset();
    if (mounted.current) { setActive(false); setRequesting(false); setResult(null); }
  }, []);
  const clearPhoto = useCallback(() => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = null;
    if (mounted.current) setPhoto(null);
  }, []);

  useEffect(() => {
    mounted.current = true;
    const instance = new InferenceEngine((next) => {
      if (mounted.current && modeRef.current !== "demo") setResult(next);
    }, (next) => {
      if (mounted.current) {
        setState(next);
        if (next.status !== "ready") setResult(null);
      }
    });
    engine.current = instance;
    void instance.initialize();
    const visibility = () => { if (document.hidden) stop(); };
    document.addEventListener("visibilitychange", visibility);
    return () => {
      mounted.current = false;
      stop(); clearPhoto(); instance.dispose(); engine.current = null;
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [clearPhoto, stop]);

  useEffect(() => {
    const container = stage.current;
    const preview = video.current;
    if (!container || !preview) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setStageRatio(width / height);
    });
    const resizeVideo = () => {
      if (modeRef.current !== "camera" || !stream.current || !preview.videoWidth || !preview.videoHeight) return;
      setRatio(preview.videoWidth / preview.videoHeight);
      // Camera rotation invalidates coordinates from the previous frame geometry.
      engine.current?.reset();
      setResult(null);
    };
    observer.observe(container);
    preview.addEventListener("resize", resizeVideo);
    return () => {
      observer.disconnect();
      preview.removeEventListener("resize", resizeVideo);
    };
  }, []);

  useEffect(() => {
    if (state.status !== "ready" || mode === "demo" || (!active && !photo)) return;
    const timer = setInterval(() => {
      const source = mode === "camera" ? video.current : image.current;
      if (document.hidden || !source) return;
      if (source instanceof HTMLVideoElement && source.readyState < 2) return;
      if (source instanceof HTMLImageElement && (!source.complete || !source.naturalWidth)) return;
      void engine.current?.processFrame(source);
    }, 250);
    return () => clearInterval(timer);
  }, [state.status, mode, active, photo]);

  function changeMode(next: Mode) {
    stop(); clearPhoto(); setError(""); setRatio(4 / 3);
    modeRef.current = next; setMode(next);
  }
  async function startCamera(selected = deviceId) {
    stop(); clearPhoto(); setError(""); modeRef.current = "camera"; setMode("camera");
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setError("Kamera memerlukan HTTPS atau localhost dan browser yang mendukung akses kamera."); return;
    }
    const current = ++operation.current;
    setRequesting(true);
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { ...(selected ? { deviceId: { exact: selected } } : { facingMode: { ideal: "environment" } }),
          width: { ideal: 1280 }, height: { ideal: 960 } }, audio: false,
      });
      if (!mounted.current || current !== operation.current) { media.getTracks().forEach((track) => track.stop()); return; }
      stream.current = media;
      media.getVideoTracks().forEach((track) => track.addEventListener("ended", () => {
        if (!mounted.current || current !== operation.current || stream.current !== media) return;
        stop();
        setError("Kamera terputus atau izinnya dicabut. Aktifkan kembali kamera untuk melanjutkan.");
      }, { once: true }));
      if (!video.current) throw new Error("Pratinjau kamera tidak tersedia.");
      video.current.srcObject = media;
      await video.current.play();
      if (!mounted.current || current !== operation.current) return;
      setRatio(video.current.videoWidth / video.current.videoHeight || 4 / 3);
      setActive(true); setRequesting(false);
      const available = await navigator.mediaDevices.enumerateDevices();
      if (mounted.current && current === operation.current) {
        setDevices(available.filter((item) => item.kind === "videoinput"));
        setDeviceId(media.getVideoTracks()[0]?.getSettings().deviceId ?? "");
      }
    } catch (cause) {
      if (!mounted.current || current !== operation.current) return;
      stop();
      const name = cause instanceof DOMException ? cause.name : "";
      setError(name === "NotAllowedError" ? "Izin kamera belum diberikan. Izinkan kamera melalui pengaturan browser, lalu coba kembali."
        : name === "NotFoundError" ? "Tidak ada kamera yang ditemukan. Anda tetap bisa memilih foto lokal."
        : name === "NotReadableError" ? "Kamera sedang dipakai aplikasi lain. Tutup aplikasi tersebut lalu coba kembali."
        : cause instanceof Error ? cause.message : "Kamera tidak dapat dibuka.");
    }
  }
  async function loadPhoto(file: File | undefined) {
    if (!file) return;
    changeMode("photo");
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type) || file.size === 0 || file.size > 10 * 1024 * 1024) {
      setError("Pilih gambar JPG, PNG, atau WebP yang valid, maksimal 10 MB."); return;
    }
    const current = ++operation.current;
    const url = URL.createObjectURL(file);
    objectUrl.current = url;
    try {
      const check = new window.Image(); check.src = url;
      await check.decode();
      if (!mounted.current || current !== operation.current) return;
      if (!check.naturalWidth || !check.naturalHeight || check.naturalWidth * check.naturalHeight > 32_000_000) {
        throw new Error("Resolusi gambar terlalu besar. Gunakan gambar maksimal 32 megapiksel.");
      }
      setRatio(check.naturalWidth / check.naturalHeight); setPhoto(url);
    } catch (cause) {
      if (!mounted.current || current !== operation.current) return;
      clearPhoto(); setError(cause instanceof Error && cause.message.includes("megapiksel") ? cause.message : "Gambar tidak dapat dibaca. Pilih berkas gambar asli yang tidak rusak.");
    }
  }
  const count = mode === "demo" ? demoObjects.length : result?.objects.length ?? 0;
  const surfaceStyle = fitPreview(ratio, stageRatio);
  const sourceLabel = mode === "demo" ? "Demo tampilan" : mode === "photo" ? "Foto lokal"
    : requesting ? "Menunggu izin kamera" : active ? "Kamera aktif" : "Kamera nonaktif";
  const emptyMessage = state.status === "ready"
    ? active || photo ? "Belum ada objek. Letakkan rimpang utuh di dalam bingkai." : "Aktifkan kamera atau pilih foto untuk mulai mengenali."
    : state.status === "loading" ? "Menunggu model siap. Kamera dan foto tetap bisa dipratinjau."
    : "Belum ada prediksi. Kamera dan foto hanya pratinjau sampai model berhasil dimuat.";

  return <div className="scanner-grid">
    <section className="scanner-workspace" aria-label="Kamera dan foto lokal">
      <div className="workspace-toolbar">
        <span className="workspace-source" role="status"><span className={`source-dot${active ? " source-dot-active" : ""}`} />{sourceLabel}</span>
        {active && devices.length > 1
          ? <select className="camera-select" aria-label="Pilih kamera" value={deviceId} onChange={(event) => { setDeviceId(event.target.value); void startCamera(event.target.value); }}>{devices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Kamera ${index + 1}`}</option>)}</select>
          : <span className="workspace-limit">Maks. 5 objek</span>}
      </div>
      {mode === "demo" && <div className="demo-banner" role="status"><Icon name="info" size={15} />Demo tampilan — bukan hasil identifikasi model</div>}
      <div ref={stage} className={`scanner-stage${mode !== "demo" && !active && !photo ? " scanner-stage-idle" : ""}`}>
        <div className="frame-surface" style={surfaceStyle}>
          <video ref={video} hidden={mode !== "camera" || !active} autoPlay playsInline muted aria-label="Pratinjau kamera lokal" />
          {mode === "photo" && photo && <Image ref={image} src={photo} alt="Pratinjau foto yang dipilih dari perangkat" width={Math.round(ratio * 1000)} height={1000} unoptimized />}
          {mode !== "demo" && result?.objects.map((object) => <div key={object.trackId} className="object-box" style={position(object.box)}><span>{object.trackId}. {object.classId !== null ? plants[object.classId]?.name : object.status === "uncertain" ? "Belum yakin" : "Menganalisis"}</span></div>)}
        </div>
        {mode !== "demo" && !active && !photo && <div className="scanner-placeholder">
          <span className="camera-symbol"><Icon name={mode === "camera" ? "camera" : "upload"} size={28} /></span>
          <h2>{requesting ? "Izinkan kamera di browser" : mode === "camera" ? "Kamera siap saat Anda siap." : "Pilih foto rimpang"}</h2>
          <p>{requesting ? "Menunggu izin. Anda bisa membatalkan kapan saja." : mode === "camera" ? "Tekan Aktifkan kamera di bawah, lalu arahkan ke rimpang." : "Gunakan JPG, PNG, atau WebP dari perangkat Anda."}</p>
        </div>}
        {mode === "demo" && <div className="demo-scene">{demoObjects.map(({ plant, box }) => <div key={plant.id} className="demo-specimen" style={position(box)}><Image src={plant.image} alt={`Contoh ${plant.name}`} fill sizes="(max-width: 800px) 35vw, 250px" /></div>)}{demoObjects.map(({ plant, box }, i) => <div key={plant.id} className="object-box" style={position(box)}><span>{i + 1}. {plant.name} · contoh</span></div>)}<span className="demo-scene-note">Anotasi ilustrasi pada foto contoh, bukan keluaran model.</span></div>}
      </div>
      <div className="workspace-bottom"><div className="scanner-controls">
        <button className={`button camera-action ${active || requesting ? "button-dark" : "button-primary"}`} onClick={() => active || requesting ? stop() : void startCamera()}><Icon name={active || requesting ? "stop" : "camera"} size={19} />{requesting ? "Batalkan" : active ? "Hentikan kamera" : "Aktifkan kamera"}</button>
        <label className="button button-light file-label"><Icon name="upload" size={18} />{photo ? "Ganti foto" : "Pilih foto"}<input className="file-input" type="file" accept="image/jpeg,image/png,image/webp" aria-label="Pilih foto rimpang" onChange={(event) => { void loadPhoto(event.target.files?.[0]); event.target.value = ""; }} /></label>
        {photo && <button className="icon-button" onClick={() => changeMode("photo")} aria-label="Hapus foto"><Icon name="close" size={16} /></button>}
      </div>
        {error && <div className="error-notice" role="alert">{error}</div>}
        <p className="scanner-small-note">{mode === "demo" ? "Demo tidak menggunakan kamera atau foto Anda." : "Foto: maks. 10 MB / 32 MP · Diproses di perangkat"}</p>
      </div>
    </section>
    <aside className="scanner-sidebar" aria-label="Status model dan hasil">
    <div className="model-notice" data-status={state.status}><Icon name={state.status === "ready" ? "check" : "info"} size={18} /><div>
      <div role="status"><h2>{state.status === "ready" ? "Model aktif di perangkat" : state.status === "loading" ? "Menyiapkan model" : state.status === "error" ? "Model gagal dimuat" : "Pratinjau saja"}</h2><p>{state.message}</p></div>
      <button className="retry-button" disabled={state.status === "loading"} onClick={() => { setResult(null); void engine.current?.initialize(); }}>Periksa kembali</button>
    </div></div>
    <section className="result-panel" aria-labelledby="results-title"><div className="result-panel-heading"><h2 id="results-title">{mode === "demo" ? "Hasil ilustrasi" : "Hasil pengenalan"}</h2><span className="count-badge" aria-live="polite">{count} / 5 objek</span></div>
      {!count && <div className="result-empty"><Icon name="scan" size={22} /><p>{emptyMessage}</p></div>}
      {mode === "demo" ? demoObjects.map(({ plant }, index) => <div className="result-item" key={plant.id}><div className="result-thumb"><Image src={plant.image} alt="" fill sizes="40px" /></div><div><h3>{index + 1}. {plant.name}</h3><p>Contoh · bukan prediksi</p></div><Link href={`/jelajah/${plant.slug}`} aria-label={`Pelajari ${plant.name}`}><Icon name="diagonal" size={16} /></Link></div>)
        : result?.objects.map((object) => <div className="result-item" key={object.trackId}><div><h3>{object.classId !== null ? plants[object.classId]?.name : object.status === "uncertain" ? "Objek belum dikenali" : "Menganalisis objek"}</h3><p>Objek {object.trackId}{object.confidence !== null ? ` · Skor model ${(object.confidence * 100).toFixed(0)}%` : " · Tunggu hasil stabil"}</p></div>{object.classId !== null && <Link href={`/jelajah/${plants[object.classId].slug}`} aria-label={`Pelajari ${plants[object.classId].name}`}><Icon name="diagonal" size={16} /></Link>}</div>)}
      {mode !== "demo" && result?.overflow && <p className="scanner-small-note">Maksimal 5 objek. Kurangi jumlah rimpang dalam bingkai.</p>}
    </section>
    <button className="scanner-demo-button" aria-pressed={mode === "demo"} onClick={() => changeMode(mode === "demo" ? "camera" : "demo")}>{mode === "demo" ? "Tutup demo" : "Lihat demo · ilustrasi"}<Icon name={mode === "demo" ? "close" : "arrow"} size={17} /></button>
    <details className="scanner-tip"><summary><Icon name="sun" size={18} />Tips pemindaian<span aria-hidden="true">+</span></summary><p>Pisahkan rimpang di permukaan polos dengan cahaya merata. Jaga kamera stabil dan seluruh objek tetap di dalam bingkai.</p><Link href="/panduan" className="text-link">Panduan & privasi<Icon name="arrow" size={16} /></Link></details>
  </aside></div>;
}

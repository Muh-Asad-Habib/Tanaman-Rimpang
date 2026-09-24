"use client";

import Image from "next/image";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { InferenceEngine, type EngineState } from "@/lib/inference/engine";
import type { FrameResult } from "@/lib/inference/contracts";
import { plants } from "@/lib/catalog";
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
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [state, setState] = useState<EngineState>({ status: "loading", message: "Memeriksa ketersediaan model..." });
  const [result, setResult] = useState<FrameResult | null>(null);
  const video = useRef<HTMLVideoElement>(null);
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
    }, (next) => { if (mounted.current) setState(next); });
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
  const surfaceStyle = { width: ratio >= 4 / 3 ? "100%" : `${ratio / (4 / 3) * 100}%`,
    height: ratio >= 4 / 3 ? `${(4 / 3) / ratio * 100}%` : "100%" };

  return <><div className="scanner-grid"><div>
    <div className="scanner-workspace"><div className="workspace-toolbar"><div className="scanner-tabs">
      <button className="scanner-tab" aria-pressed={mode === "camera"} onClick={() => changeMode("camera")}><Icon name="camera" size={16} />Kamera</button>
      <button className="scanner-tab" aria-pressed={mode === "photo"} onClick={() => changeMode("photo")}><Icon name="upload" size={16} />Foto lokal</button>
    </div><span className="eyebrow" style={{ fontSize: 8 }}>RUANG PINDAI</span></div>
      {mode === "demo" && <div className="demo-banner" role="status"><Icon name="info" size={15} />Demo tampilan — bukan hasil identifikasi model</div>}
      <div className="scanner-stage">
        <div className="frame-surface" style={surfaceStyle}>
          <video ref={video} hidden={mode !== "camera" || !active} autoPlay playsInline muted aria-label="Pratinjau kamera lokal" />
          {mode === "photo" && photo && <Image ref={image} src={photo} alt="Pratinjau foto yang dipilih dari perangkat" width={Math.round(ratio * 1000)} height={1000} unoptimized />}
          {mode !== "demo" && result?.objects.map((object) => <div key={object.trackId} className="object-box" style={position(object.box)}><span>{object.classId !== null ? plants[object.classId]?.name : object.status === "uncertain" ? "Belum yakin" : "Menganalisis"}</span></div>)}
        </div>
        {mode !== "demo" && !active && !photo && <div className="scanner-placeholder"><div className="viewfinder" /><span className="camera-symbol"><Icon name={mode === "camera" ? "camera" : "upload"} size={30} /></span><h2>{requesting ? "Menghubungkan kamera..." : mode === "camera" ? "Mulai dari satu rimpang." : "Lihat lebih dekat."}</h2><p>{requesting ? "Berikan izin kamera pada browser Anda." : mode === "camera" ? "Aktifkan kamera dan arahkan pada rimpang yang ingin Anda kenali." : "Pilih foto rimpang dari perangkat. Foto tidak dikirim ke server."}</p><span className="stage-badge">PRATINJAU LOKAL · PRIVASI TERJAGA</span></div>}
        {mode === "demo" && <div className="demo-scene">{demoObjects.map(({ plant, box }) => <div key={plant.id} className="demo-specimen" style={position(box)}><Image src={plant.image} alt={`Contoh ${plant.name}`} fill sizes="(max-width: 800px) 35vw, 250px" /></div>)}{demoObjects.map(({ plant, box }, i) => <div key={plant.id} className="object-box" style={position(box)}><span>{i + 1}. {plant.name} · contoh</span></div>)}<span className="demo-scene-note">Anotasi ilustrasi pada foto contoh, bukan keluaran model.</span></div>}
      </div>
      <div className="workspace-bottom"><div className="scanner-controls">
        {mode === "camera" && <button className={`button ${active || requesting ? "button-outline" : "button-dark"}`} onClick={() => active || requesting ? stop() : void startCamera()}><Icon name={active || requesting ? "stop" : "camera"} size={16} />{requesting ? "Batalkan" : active ? "Hentikan kamera" : "Aktifkan kamera"}</button>}
        {mode === "demo" && <button className="button button-dark" onClick={() => changeMode("camera")}><Icon name="camera" size={16} />Kembali ke kamera</button>}
        <label className="button button-light file-label"><Icon name="upload" size={16} />{photo ? "Ganti foto" : "Pilih foto"}<input className="file-input" type="file" accept="image/jpeg,image/png,image/webp" aria-label="Pilih foto rimpang" onChange={(event) => { void loadPhoto(event.target.files?.[0]); event.target.value = ""; }} /></label>
        {photo && <button className="icon-button" onClick={() => changeMode("photo")} aria-label="Hapus foto"><Icon name="close" size={16} /></button>}
      </div>{active && devices.length > 1 && <select className="camera-select" aria-label="Pilih kamera" value={deviceId} onChange={(e) => { setDeviceId(e.target.value); void startCamera(e.target.value); }}>{devices.map((device, i) => <option key={device.deviceId} value={device.deviceId}>{device.label || `Kamera ${i + 1}`}</option>)}</select>}
        <p className="scanner-small-note">{mode === "demo" ? "Mode demo tidak menggunakan kamera maupun foto Anda." : "JPG, PNG, WebP · Maks. 10 MB · Tidak diunggah ke server"}</p>
      </div>
    </div>{error && <div className="error-notice" role="alert">{error}</div>}
  </div><aside className="scanner-sidebar">
    <div className="model-notice" role="status"><Icon name={state.status === "ready" ? "check" : "info"} size={19} /><div><h2>{state.status === "ready" ? "Model tersedia di perangkat" : state.status === "loading" ? "Memeriksa model" : state.status === "error" ? "Model belum dapat dimuat" : "Model belum tersedia"}</h2><p>{state.message}</p>{state.status === "unavailable" && <p>Kamera dapat digunakan sebagai pratinjau. Belum ada prediksi nyata pada versi ini.</p>}<button className="retry-button" disabled={state.status === "loading"} onClick={() => { setResult(null); void engine.current?.initialize(); }}>Periksa kembali</button></div></div>
    <div className="result-panel"><div className="result-panel-heading"><h2>{mode === "demo" ? "Contoh tampilan hasil" : "Hasil pengenalan"}</h2><span className="count-badge">{count} / 5 objek</span></div>
      {!count && <div className="result-empty"><Icon name="leaf" size={35} /><p>{state.status === "ready" && (active || photo) ? "Arahkan pada rimpang. Objek yang dikenali akan muncul di sini." : "Belum ada hasil. Sambil menunggu model, Anda bisa menjelajahi katalog atau melihat demo."}</p></div>}
      {mode === "demo" ? demoObjects.map(({ plant }) => <div className="result-item" key={plant.id}><div className="result-thumb"><Image src={plant.image} alt="" fill sizes="48px" /></div><div><h3>{plant.name}</h3><p>Anotasi contoh · bukan prediksi</p></div><Link href={`/jelajah/${plant.slug}`} aria-label={`Pelajari ${plant.name}`}><Icon name="diagonal" size={16} /></Link></div>)
        : result?.objects.map((object) => <div className="result-item" key={object.trackId}><div><h3>{object.classId !== null ? plants[object.classId]?.name : object.status === "uncertain" ? "Objek belum dikenali" : "Menganalisis objek"}</h3><p>Objek {object.trackId}{object.confidence !== null ? ` · Skor model ${(object.confidence * 100).toFixed(0)}%` : " · Tunggu hasil stabil"}</p></div>{object.classId !== null && <Link href={`/jelajah/${plants[object.classId].slug}`} aria-label={`Pelajari ${plants[object.classId].name}`}><Icon name="diagonal" size={16} /></Link>}</div>)}
      {mode !== "demo" && result?.overflow && <p className="scanner-small-note">Maksimal 5 objek. Kurangi jumlah rimpang dalam bingkai.</p>}
    </div>
    {mode !== "demo" && <button className="button button-outline scanner-demo-button" onClick={() => changeMode("demo")}>Lihat demo tampilan <Icon name="arrow" size={17} /></button>}
    <div className="scanner-tip"><Icon name="sun" size={19} /><p>Letakkan rimpang terpisah di permukaan polos. Cahaya yang merata membantu kamera melihat bentuknya.</p></div>
  </aside></div><div className="scanner-bottom-note"><Icon name="shield" size={21} /><span>Foto tetap milik Anda. Tidak ada unggahan gambar maupun rekaman kamera.</span><Link href="/panduan">Baca panduan</Link></div></>;
}

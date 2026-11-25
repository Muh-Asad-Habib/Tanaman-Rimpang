"use client";

import { useSyncExternalStore } from "react";
import { plants } from "@/lib/catalog";
import { PlantCard } from "./plant-card";
import { Icon } from "./icons";

const categories = ["Semua rimpang", ...new Set(plants.map((plant) => plant.family))];
const urlChange = "catalog-url-change";

function subscribe(onChange: () => void) {
  window.addEventListener("popstate", onChange);
  window.addEventListener(urlChange, onChange);
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(urlChange, onChange);
  };
}

function writeParams(update: (params: URLSearchParams) => void) {
  const params = new URLSearchParams(window.location.search);
  update(params);
  const search = params.toString();
  window.history.replaceState(window.history.state, "", `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`);
  window.dispatchEvent(new Event(urlChange));
}

export function CatalogGrid() {
  const search = useSyncExternalStore(subscribe, () => window.location.search, () => "");
  const params = new URLSearchParams(search);
  const query = params.get("q") ?? "";
  const group = params.get("kelompok");
  const category = group && categories.includes(group) ? group : categories[0];
  const setQuery = (value: string) => writeParams((next) => { if (value) next.set("q", value); else next.delete("q"); });
  const setCategory = (value: string) => writeParams((next) => {
    if (value !== categories[0]) next.set("kelompok", value); else next.delete("kelompok");
  });
  const matches = plants.filter((plant) => (category === categories[0] || plant.family === category)
    && `${plant.name} ${plant.scientific}`.toLowerCase().includes(query.toLowerCase().trim()));
  return <><div className="catalog-tools"><div className="filter-list" role="group" aria-label="Kelompok rimpang">{categories.map((item) =>
    <button key={item} type="button" className={`filter ${item === category ? "selected" : ""}`} aria-pressed={item === category} onClick={() => setCategory(item)}>{item}</button>)}</div>
    <label className="search-field"><Icon name="search" size={19} /><input type="search" name="q" aria-label="Cari rimpang" placeholder="Cari nama rimpang…"
      autoComplete="off" spellCheck={false} enterKeyHint="search" value={query} onChange={(e) => setQuery(e.target.value)} /></label></div>
    <p className="result-count" aria-live="polite">{matches.length} jenis untuk dijelajahi</p>
    <div className="catalog-grid">{matches.map((plant, index) => <PlantCard key={plant.id} plant={plant} index={index} eager={index < 8} highPriority={index < 2} />)}</div>
    {!matches.length && <div className="empty-state"><Icon name="search" size={30} /><h3>Belum ada yang cocok.</h3><p>Coba nama yang lain atau tampilkan semua jenis.</p><button type="button" className="button button-outline" onClick={() => { setQuery(""); setCategory(categories[0]); }}>Tampilkan semua</button></div>}
  </>;
}

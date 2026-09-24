"use client";

import { useState } from "react";
import { plants } from "@/lib/catalog";
import { PlantCard } from "./plant-card";
import { Icon } from "./icons";

export function CatalogGrid() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("Semua rimpang");
  const categories = ["Semua rimpang", ...new Set(plants.map((plant) => plant.family))];
  const matches = plants.filter((plant) => (category === categories[0] || plant.family === category)
    && `${plant.name} ${plant.scientific}`.toLowerCase().includes(query.toLowerCase().trim()));
  return <><div className="catalog-tools"><div className="filter-list" aria-label="Kelompok rimpang">{categories.map((item) =>
    <button key={item} className={`filter ${item === category ? "selected" : ""}`} aria-pressed={item === category} onClick={() => setCategory(item)}>{item}</button>)}</div>
    <label className="search-field"><Icon name="search" size={19} /><input aria-label="Cari rimpang" placeholder="Cari nama rimpang..." value={query} onChange={(e) => setQuery(e.target.value)} /></label></div>
    <p className="result-count" aria-live="polite">{matches.length} jenis untuk dijelajahi</p>
    <div className="catalog-grid">{matches.map((plant, index) => <PlantCard key={plant.id} plant={plant} index={index} />)}</div>
    {!matches.length && <div className="empty-state"><Icon name="search" size={30} /><h3>Belum ada yang cocok.</h3><p>Coba nama yang lain atau tampilkan semua jenis.</p><button className="button button-outline" onClick={() => { setQuery(""); setCategory(categories[0]); }}>Tampilkan semua</button></div>}
  </>;
}

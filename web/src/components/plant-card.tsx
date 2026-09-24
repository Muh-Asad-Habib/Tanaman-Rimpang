import Image from "next/image";
import Link from "next/link";
import type { Plant } from "@/lib/catalog";
import { Icon } from "./icons";

export function PlantCard({ plant, index = 0 }: { plant: Plant; index?: number }) {
  return <Link href={`/jelajah/${plant.slug}`} className="plant-card">
    <div className="plant-card-art" style={{ background: plant.color }}>
      <span className="specimen-number">SPESIMEN {String(plant.id + 1).padStart(2, "0")}</span>
      <Image src={plant.image} alt={`Rimpang ${plant.name.toLowerCase()}`} fill sizes="(max-width: 580px) 90vw, (max-width: 1000px) 45vw, 25vw" className={`plant-cutout tilt-${index % 3}`} />
      <span className="plant-card-link"><Icon name="diagonal" /></span>
    </div><div className="plant-card-heading"><h3>{plant.name}</h3><span>{plant.family.split(" & ")[0]}</span></div>
    <p className="scientific">{plant.scientific.replace(" (nama rujukan)", "")}</p>
  </Link>;
}

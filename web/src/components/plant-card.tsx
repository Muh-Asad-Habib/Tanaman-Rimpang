import Image from "next/image";
import Link from "next/link";
import type { Plant } from "@/lib/catalog";
import { Icon } from "./icons";

export function PlantCard({ plant, index = 0, eager = false }: { plant: Plant; index?: number; eager?: boolean }) {
  return <Link href={`/jelajah/${plant.slug}`} className="plant-card">
    <div className="plant-card-art" style={{ background: plant.color }}>
      <span className="specimen-number">SPESIMEN {String(plant.id + 1).padStart(2, "0")}</span>
      <Image src={plant.image} alt={`Rimpang ${plant.name.toLowerCase()}`} fill loading={eager ? "eager" : "lazy"} sizes="(max-width: 800px) 50vw, (max-width: 1100px) 33vw, (max-width: 1352px) 25vw, 300px" className={`plant-cutout tilt-${index % 3}`} />
      <span className="plant-card-link"><Icon name="diagonal" /></span>
    </div><div className="plant-card-heading"><h3>{plant.name}</h3><span>{plant.family.split(" & ")[0]}</span></div>
    <p className="scientific">{plant.scientific.replace(" (nama rujukan)", "")}</p>
  </Link>;
}

import type { CSSProperties } from "react";

const paths = {
  arrow: "M4 12h15M13 5l7 7-7 7",
  diagonal: "M6 18 18 6M6 6h12v12",
  camera: "M4 7h4l2-3h4l2 3h4v13H4V7Z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  leaf: "M19 4C9 3 3 7 5 14s12 7 14-10Z M5 20 15 9",
  search: "M20 20l-5-5M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z",
  upload: "M12 16V3M7 8l5-5 5 5M4 15v6h16v-6",
  shield: "m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3ZM8 12l3 3 5-6",
  close: "m6 6 12 12M6 18 18 6",
  menu: "M4 7h16M4 12h16M4 17h16",
  sun: "M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1 1m12 12 1 1M5 19l1-1M18 6l1-1M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z",
  scan: "M3 9V3h6m6 0h6v6M3 15v6h6m6 0h6v-6M7 12h10",
  swap: "M4 7h15l-4-4M20 17H5l4 4",
  info: "M12 11v6M12 7v.2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z",
  book: "M12 6c-4-3-8-2-10-1v15c4-2 7-2 10 0 3-2 6-2 10 0V5c-2-1-6-2-10 1Zm0 0v14",
  check: "m5 12 4 4L19 6",
  stop: "M6 6h12v12H6z",
} as const;

export function Icon({ name, size = 20, style, className }: {
  name: keyof typeof paths; size?: number; style?: CSSProperties; className?: string;
}) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"
    style={style} className={className}><path d={paths[name]} /></svg>;
}

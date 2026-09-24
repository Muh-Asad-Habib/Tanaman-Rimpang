"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Icon } from "./icons";

const links = [{ href: "/", text: "Beranda" }, { href: "/jelajah", text: "Jelajahi rimpang" }, { href: "/panduan", text: "Panduan" }];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return <header className="site-header">
    <div className="container header-inner">
      <Link href="/" className="brand" aria-label="Rimpang, beranda" onClick={() => setOpen(false)}>
        <span className="brand-mark"><Icon name="leaf" size={23} /></span>rimpang<span className="brand-dot">.</span>
      </Link>
      <nav className="desktop-nav" aria-label="Navigasi utama">
        {links.map((item) => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? "page" : undefined}>{item.text}</Link>)}
      </nav>
      <Link href="/pindai" className="button button-dark header-cta"><Icon name="scan" size={17} /> Mulai pindai <Icon name="diagonal" size={16} /></Link>
      <button className="mobile-menu icon-button" aria-label={open ? "Tutup menu" : "Buka menu"}
        aria-expanded={open} aria-controls="mobile-navigation" onClick={() => setOpen(!open)}><Icon name={open ? "close" : "menu"} /></button>
    </div>
    {open && <nav id="mobile-navigation" className="mobile-nav" aria-label="Navigasi mobile">
      {[...links, { href: "/pindai", text: "Mulai pindai" }].map((item) => <Link key={item.href} href={item.href}
        aria-current={pathname === item.href ? "page" : undefined} onClick={() => setOpen(false)}>{item.text}<Icon name="arrow" size={16} /></Link>)}
    </nav>}
  </header>;
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Icon } from "./icons";

const links = [{ href: "/", text: "Beranda" }, { href: "/jelajah", text: "Jelajahi rimpang" }, { href: "/panduan", text: "Panduan" }];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const header = useRef<HTMLElement>(null);
  const toggle = useRef<HTMLButtonElement>(null);
  const currentPage = (href: string) => pathname === href ? "page"
    : href !== "/" && pathname.startsWith(`${href}/`) ? "location" : undefined;

  useEffect(() => {
    const compact = window.matchMedia("(max-width: 900px)");
    const close = () => setOpen(false);
    compact.addEventListener("change", close);
    window.addEventListener("popstate", close);
    return () => {
      compact.removeEventListener("change", close);
      window.removeEventListener("popstate", close);
    };
  }, []);

  useEffect(() => {
    if (!open) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setOpen(false);
      toggle.current?.focus();
    };
    const outside = (event: Event) => {
      if (event.target instanceof Node && !header.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("keydown", escape);
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => {
      document.removeEventListener("keydown", escape);
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
    };
  }, [open]);

  return <header ref={header} className="site-header">
    <div className="container header-inner">
      <Link href="/" className="brand" aria-label="Rimpang, beranda" onClick={() => setOpen(false)}>
        <span className="brand-mark"><Icon name="leaf" size={23} /></span>rimpang<span className="brand-dot">.</span>
      </Link>
      <nav className="desktop-nav" aria-label="Navigasi utama">
        {links.map((item) => <Link key={item.href} href={item.href} aria-current={currentPage(item.href)}>{item.text}</Link>)}
      </nav>
      <Link href="/pindai" className="button button-dark header-cta" onClick={() => setOpen(false)}><Icon name="scan" size={17} /> Mulai pindai <Icon name="diagonal" size={16} /></Link>
      <button ref={toggle} className="mobile-menu icon-button" aria-label={open ? "Tutup menu" : "Buka menu"}
        aria-expanded={open} aria-controls="mobile-navigation" onClick={() => setOpen(!open)}><Icon name={open ? "close" : "menu"} /></button>
    </div>
    <nav id="mobile-navigation" className="mobile-nav" aria-label="Navigasi mobile" hidden={!open}>
      {[...links, { href: "/pindai", text: "Mulai pindai" }].map((item) => <Link key={item.href} href={item.href}
        aria-current={currentPage(item.href)} onClick={() => setOpen(false)}>{item.text}<Icon name="arrow" size={16} /></Link>)}
    </nav>
  </header>;
}

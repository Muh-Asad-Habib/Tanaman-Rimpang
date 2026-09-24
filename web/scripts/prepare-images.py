"""Create local, optimized editorial assets; never modify source photographs."""
import argparse
from pathlib import Path

from PIL import Image, ImageFilter

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for slug in ("jahe", "jahe-merah", "kencur", "kunyit", "kunyit-putih", "lempuyang",
             "lengkuas", "temu-hitam", "temu-kunci", "temulawak"):
    source = args.source / (slug.replace("-", "_") + ".jpg")
    with Image.open(source) as original:
        photo = original.convert("RGB")
        photo.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
        photo.save(args.output / f"{slug}-photo.webp", quality=84)
        pixels = list(photo.getdata())
        alpha = Image.new("L", photo.size)
        # Remove the blue studio backdrop only for decorative cutouts, not training data.
        alpha.putdata([
            max(0, min(255, round(255 - max(0, b - (r + g) / 2 - 4) * 16)))
            for r, g, b in pixels
        ])
        alpha = alpha.filter(ImageFilter.GaussianBlur(0.45))
        cutout = photo.convert("RGBA")
        cutout.putalpha(alpha)
        bounds = alpha.point(lambda p: 255 if p > 120 else 0).getbbox()
        if not bounds:
            raise ValueError(f"No foreground found: {source}")
        cutout = cutout.crop(bounds)
        cutout.save(args.output / f"{slug}.webp", quality=88)
        print(f"{slug}: {photo.size}, cutout {cutout.size}")

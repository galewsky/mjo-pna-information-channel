#!/usr/bin/env python3
"""
Plot split-domain map with labels for domain IDs (0..71).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle


WIDTH = 10
LAT_MIN, LAT_MAX = -25, 25
LON_MIN, LON_MAX = 0, 360


def get_domain_rect(domain_id: int):
    domain_id = int(domain_id)
    if 0 <= domain_id < 36:
        lon_start = domain_id * WIDTH
        lat_start = -20
        height = 20
    elif 36 <= domain_id < 72:
        lon_start = (domain_id - 36) * WIDTH
        lat_start = 0
        height = 20
    else:
        return None
    return lon_start, lat_start, WIDTH, height


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("domain_map_labels.png"))
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f7ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color="black")
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    ax.plot([0, 360], [0, 0], color="black", linewidth=1.0, transform=ccrs.PlateCarree())

    for domain_id in range(72):
        rect = get_domain_rect(domain_id)
        if rect is None:
            continue
        x, y, w, h = rect
        ax.add_patch(Rectangle((x, y), w, h,
                               transform=ccrs.PlateCarree(),
                               facecolor="none",
                               edgecolor="black",
                               linewidth=0.6))
        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy, str(domain_id),
                transform=ccrs.PlateCarree(),
                ha="center", va="center",
                fontsize=7, color="black")

    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

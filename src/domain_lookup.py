#!/usr/bin/env python3
"""
Domain lookup helper for flux_meridional-style split domains.

Domains 0-35: lat -20..0, lon 10-degree bins (0..360)
Domains 36-71: lat 0..20, lon 10-degree bins (0..360)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


WIDTH_DEG = 10
LAT_BANDS = [(-20, 0), (0, 20)]


def domain_bounds(domain_id: int) -> Tuple[int, int, int, int]:
    domain_id = int(domain_id)
    if 0 <= domain_id < 36:
        lon_start = domain_id * WIDTH_DEG
        lat_start, lat_end = LAT_BANDS[0]
    elif 36 <= domain_id < 72:
        lon_start = (domain_id - 36) * WIDTH_DEG
        lat_start, lat_end = LAT_BANDS[1]
    else:
        raise ValueError(f"Domain id {domain_id} outside 0..71.")
    lon_end = lon_start + WIDTH_DEG
    return lon_start, lon_end, lat_start, lat_end


def to_west(lon: int) -> int:
    return lon - 360 if lon > 180 else lon


def build_rows(domains: Iterable[int]) -> pd.DataFrame:
    rows = []
    for d in domains:
        lon0, lon1, lat0, lat1 = domain_bounds(int(d))
        rows.append({
            "domain_id": int(d),
            "lon_start_e": lon0,
            "lon_end_e": lon1,
            "lat_start": lat0,
            "lat_end": lat1,
            "lon_start_w": to_west(lon0),
            "lon_end_w": to_west(lon1),
        })
    return pd.DataFrame(rows).sort_values("domain_id")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("domains", nargs="+", type=int, help="Domain ids (0..71)")
    p.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = build_rows(args.domains)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

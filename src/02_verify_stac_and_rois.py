"""
Step 2: Verify Sentinel-1 RTC and Sentinel-2 L2A availability on Microsoft
Planetary Computer STAC for our 4 candidate Sicilian reservoirs (2021-2026),
and test our zero-GDAL pure-Python COG reader on both S2 L2A and S1 RTC!
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import requests
import pystac_client
import planetary_computer

from cog_reader import bbox_lonlat_to_utm33n, read_cog_window_utm

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESERVOIRS = {
    "rosamarina": {
        "cod": "dig-22",
        "name": "Lago Rosamarina",
        "dam_lat": 37.959,
        "dam_lon": 13.646,
        "bbox": [13.595, 37.915, 13.685, 37.975],
    },
    "poma": {
        "cod": "dig-18",
        "name": "Lago Poma",
        "dam_lat": 38.012,
        "dam_lon": 13.063,
        "bbox": [13.040, 37.965, 13.115, 38.025],
    },
    "castello": {
        "cod": "dig-03",
        "name": "Lago Castello (Magazzolo)",
        "dam_lat": 37.582,
        "dam_lon": 13.420,
        "bbox": [13.395, 37.565, 13.465, 37.620],
    },
    "fanaco": {
        "cod": "dig-07",
        "name": "Lago Fanaco",
        "dam_lat": 37.661,
        "dam_lon": 13.550,
        "bbox": [13.525, 37.645, 13.585, 37.695],
    },
}

TIME_RANGE = "2021-01-01/2026-02-28"


def main() -> None:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    session = requests.Session()

    summary = {}
    for key, info in RESERVOIRS.items():
        bbox = info["bbox"]
        utm_bbox = bbox_lonlat_to_utm33n(tuple(bbox), snap_res=20.0)
        print(f"\n=== {info['name']} ({info['cod']}) ===")

        s2_items = list(
            catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=TIME_RANGE,
            ).items()
        )
        s1_items = list(
            catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=bbox,
                datetime=TIME_RANGE,
            ).items()
        )

        s2_dates = sorted({it.datetime.date().isoformat() for it in s2_items})
        s1_dates = sorted({it.datetime.date().isoformat() for it in s1_items})

        s2_2025_26 = [d for d in s2_dates if d >= "2025-01-01"]
        s1_2025_26 = [d for d in s1_dates if d >= "2025-01-01"]

        # Test pure-Python COG reader on 1 clear S2 scene and 1 S1 RTC scene
        clear_s2 = [
            it for it in s2_items
            if (it.properties.get("eo:cloud_cover", 100) < 5)
            and ("2023-04-01" <= it.datetime.date().isoformat() <= "2023-07-01")
        ]
        water_km2 = np.nan
        s1_vv_water_db = np.nan
        s1_vv_land_db = np.nan

        if clear_s2:
            t0 = time.perf_counter()
            it2 = clear_s2[0]
            g = read_cog_window_utm(it2.assets["B03"].href, utm_bbox, target_res=20.0, session=session)
            swir = read_cog_window_utm(it2.assets["B11"].href, utm_bbox, target_res=20.0, session=session)
            scl = read_cog_window_utm(it2.assets["SCL"].href, utm_bbox, target_res=20.0, nearest=True, session=session)
            mndwi = (g - swir) / (g + swir + 1e-6)
            water_mask = (mndwi > 0.0) & (scl == 6)
            water_km2 = float(water_mask.sum() * (20 * 20) / 1e6)
            dt_s2 = time.perf_counter() - t0
            print(
                f"  [S2 OK in {dt_s2:.2f}s] {it2.datetime.date()} shape={g.shape} | "
                f"water pixels={int(water_mask.sum()):,} ({water_km2:.2f} km^2)"
            )

            # Find closest S1 RTC scene within +-10 days
            t2_ts = pd.Timestamp(it2.datetime.date())
            s1_sorted = sorted(
                s1_items,
                key=lambda x: abs((pd.Timestamp(x.datetime.date()) - t2_ts).days),
            )
            if s1_sorted:
                t1 = time.perf_counter()
                it1 = s1_sorted[0]
                vv = read_cog_window_utm(it1.assets["vv"].href, utm_bbox, target_res=20.0, session=session)
                vv_db = 10.0 * np.log10(np.clip(vv, 1e-6, None))
                dt_s1 = time.perf_counter() - t1
                if water_mask.sum() > 20:
                    s1_vv_water_db = float(np.nanmedian(vv_db[water_mask]))
                    s1_vv_land_db = float(np.nanmedian(vv_db[~water_mask & (scl > 0)]))
                print(
                    f"  [S1 OK in {dt_s1:.2f}s] {it1.datetime.date()} shape={vv.shape} | "
                    f"median VV water={s1_vv_water_db:.1f} dB vs land={s1_vv_land_db:.1f} dB"
                )

        orbits = pd.Series([it.properties.get("sat:orbit_state") for it in s1_items]).value_counts().to_dict()
        print(f"  S2 unique dates (2021-2026): {len(s2_dates)} (2025-26: {len(s2_2025_26)})")
        print(f"  S1 unique dates (2021-2026): {len(s1_dates)} (2025-26: {len(s1_2025_26)}) | orbits={orbits}")

        summary[key] = {
            **info,
            "utm_bbox": list(utm_bbox),
            "s2_total_dates": len(s2_dates),
            "s2_2025_26_dates": len(s2_2025_26),
            "s1_total_dates": len(s1_dates),
            "s1_2025_26_dates": len(s1_2025_26),
            "s1_orbits": orbits,
            "sanity_water_km2_2023": round(water_km2, 3) if not np.isnan(water_km2) else None,
            "s1_vv_water_db": round(s1_vv_water_db, 2) if not np.isnan(s1_vv_water_db) else None,
            "s1_vv_land_db": round(s1_vv_land_db, 2) if not np.isnan(s1_vv_land_db) else None,
        }

    with open(OUT_DIR / "stac_inventory_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUT_DIR / 'stac_inventory_summary.json'}")


if __name__ == "__main__":
    main()

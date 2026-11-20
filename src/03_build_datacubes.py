"""
Step 3: Download and cache compact Sentinel-2 L2A and Sentinel-1 RTC datacubes
(20 m UTM Zone 33N) for the 4 Sicilian reservoirs (2022-01-01 to 2026-02-28)
using our zero-GDAL pure-Python HTTP Range COG reader.

Design:
- Training / Calibration period: 2022-01-01 to 2024-12-31
  * S2: up to 2 clearest scenes per month (~72 scenes/reservoir) for self-supervised
    water masks + monthly hypsometric area-volume calibration.
  * S1: ~2-3 scenes per month (prioritizing dates within +-3 days of clear S2 scenes
    and near the 1st of each month for monthly ground-truth alignment).
- Strict Out-of-Time Test period: 2025-01-01 to 2026-02-28
  * S2: ALL unique-date passes (to accurately quantify real cloud blindness!).
  * S1: ALL unique-date passes (to test full all-weather monitoring!).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import requests
import pystac_client
import planetary_computer

from cog_reader import bbox_lonlat_to_utm33n, read_cog_window_utm

CUBE_DIR = Path("data/cubes")
CUBE_DIR.mkdir(parents=True, exist_ok=True)

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

TIME_RANGE = "2022-01-01/2026-02-28"
RES_M = 20.0
MAX_WORKERS = 8


def _select_s2_items(items: list) -> list:
    """
    Deduplicate by date (picking lowest scene cloud cover per date).
    Keep ALL dates in 2025-2026 (test), and up to 2 clearest dates per month
    in 2022-2024 (train) so training has dense seasonal & drought coverage.
    """
    by_date = {}
    for it in items:
        d = it.datetime.date().isoformat()
        cc = float(it.properties.get("eo:cloud_cover", 100.0))
        if d not in by_date or cc < by_date[d][0]:
            by_date[d] = (cc, it)

    selected = []
    # 2022-2024: pick up to 2 clearest scenes per month (and prioritize 1st-of-month proximity if tied)
    train_by_ym: dict[str, list] = {}
    for d, (cc, it) in sorted(by_date.items()):
        if d >= "2025-01-01":
            selected.append(it)
        else:
            ym = d[:7]
            train_by_ym.setdefault(ym, []).append((cc, d, it))

    for ym, lst in sorted(train_by_ym.items()):
        lst.sort(key=lambda x: (x[0], abs(int(x[1][8:10]) - 1)))
        for cc, d, it in lst[:2]:
            selected.append(it)

    selected.sort(key=lambda it: it.datetime)
    return selected


def _select_s1_items(items: list, selected_s2_dates: list[str]) -> list:
    """
    Deduplicate S1 RTC items by (date, orbit_state).
    Keep ALL dates in 2025-2026 (test).
    In 2022-2024 (train), keep scenes within +-4 days of selected S2 dates
    PLUS the scene closest to the 1st of each month (ensuring both paired
    S1-S2 training and monthly volume calibration).
    """
    by_key = {}
    for it in items:
        d = it.datetime.date().isoformat()
        orb = it.properties.get("sat:orbit_state", "unknown")
        k = (d, orb)
        if k not in by_key:
            by_key[k] = it

    s2_ts = [pd.Timestamp(d) for d in selected_s2_dates if d < "2025-01-01"]
    month_starts = pd.date_range("2022-01-01", "2024-12-01", freq="MS")

    selected = {}
    # 1. All 2025-2026
    for (d, orb), it in by_key.items():
        if d >= "2025-01-01":
            selected[(d, orb)] = it

    # 2. 2022-2024: paired within +-4 days of S2 train dates
    for (d, orb), it in by_key.items():
        if d < "2025-01-01":
            ts = pd.Timestamp(d)
            if any(abs((ts - t2).days) <= 4 for t2 in s2_ts):
                selected[(d, orb)] = it

    # 3. 2022-2024: closest to each 1st of month
    train_items = [((d, orb), it) for (d, orb), it in by_key.items() if d < "2025-01-01"]
    for ms in month_starts:
        if train_items:
            best_k, best_it = min(
                train_items,
                key=lambda x: abs((pd.Timestamp(x[0][0]) - ms).days),
            )
            selected[best_k] = best_it

    out = list(selected.values())
    out.sort(key=lambda it: it.datetime)
    return out


def _fetch_one_s2(it, utm_bbox: tuple) -> dict | None:
    session = requests.Session()
    try:
        g = read_cog_window_utm(it.assets["B03"].href, utm_bbox, target_res=RES_M, session=session)
        swir = read_cog_window_utm(it.assets["B11"].href, utm_bbox, target_res=RES_M, session=session)
        scl = read_cog_window_utm(it.assets["SCL"].href, utm_bbox, target_res=RES_M, nearest=True, session=session)

        valid = (scl > 0) & np.isfinite(g) & np.isfinite(swir) & (g > 0)
        valid_frac = float(valid.mean())
        if valid_frac < 0.85:
            return None  # partial swath edge

        # SCL cloud/shadow/nodata classes: 0, 1, 3 (cloud shadow), 8, 9, 10 (clouds/cirrus)
        cloudy = np.isin(scl, [0, 1, 3, 8, 9, 10])
        roi_cloud_frac = float(cloudy.mean())

        mndwi = ((g - swir) / (g + swir + 1e-6)).astype(np.float32)
        return {
            "date": it.datetime.date().isoformat(),
            "id": it.id,
            "scene_cloud_pct": float(it.properties.get("eo:cloud_cover", np.nan)),
            "roi_cloud_frac": roi_cloud_frac,
            "valid_frac": valid_frac,
            "mndwi": np.clip(mndwi, -1.0, 1.0).astype(np.float16),
            "scl": scl.astype(np.uint8),
        }
    except Exception as e:
        print(f"    [WARN S2 {it.id}] {e}")
        return None


def _fetch_one_s1(it, utm_bbox: tuple) -> dict | None:
    session = requests.Session()
    try:
        if "vv" not in it.assets or "vh" not in it.assets:
            return None
        vv = read_cog_window_utm(it.assets["vv"].href, utm_bbox, target_res=RES_M, session=session)
        vh = read_cog_window_utm(it.assets["vh"].href, utm_bbox, target_res=RES_M, session=session)

        valid = np.isfinite(vv) & np.isfinite(vh) & (vv > 1e-6) & (vh > 1e-6)
        valid_frac = float(valid.mean())
        if valid_frac < 0.85:
            return None  # partial swath edge

        vv_db = np.where(valid, 10.0 * np.log10(np.clip(vv, 1e-6, None)), np.nan).astype(np.float16)
        vh_db = np.where(valid, 10.0 * np.log10(np.clip(vh, 1e-6, None)), np.nan).astype(np.float16)
        return {
            "date": it.datetime.date().isoformat(),
            "id": it.id,
            "orbit_state": it.properties.get("sat:orbit_state", "unknown"),
            "valid_frac": valid_frac,
            "vv_db": vv_db,
            "vh_db": vh_db,
        }
    except Exception as e:
        print(f"    [WARN S1 {it.id}] {e}")
        return None


def process_reservoir(key: str, info: dict) -> None:
    s2_npz = CUBE_DIR / f"{key}_s2.npz"
    s1_npz = CUBE_DIR / f"{key}_s1.npz"
    if s2_npz.exists() and s1_npz.exists():
        print(f"[SKIP] {info['name']} cubes already cached.")
        return

    t0 = time.perf_counter()
    bbox = info["bbox"]
    utm_bbox = bbox_lonlat_to_utm33n(tuple(bbox), snap_res=RES_M)
    print(f"\n=== Processing {info['name']} ({info['cod']}) | UTM bbox={utm_bbox} ===")

    # Open fresh signed catalog per reservoir so SAS tokens never expire mid-run
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    s2_raw = list(catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=TIME_RANGE).items())
    s1_raw = list(catalog.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=TIME_RANGE).items())

    s2_sel = _select_s2_items(s2_raw)
    s2_sel_dates = [it.datetime.date().isoformat() for it in s2_sel]
    s1_sel = _select_s1_items(s1_raw, s2_sel_dates)

    print(f"  Selected {len(s2_sel)} S2 scenes and {len(s1_sel)} S1 RTC scenes.")

    # Fetch S2 in parallel
    s2_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_fetch_one_s2, it, utm_bbox) for it in s2_sel]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                s2_results.append(res)
    s2_results.sort(key=lambda x: x["date"])

    # Deduplicate any identical date (keep lowest roi_cloud_frac)
    s2_dedup = {}
    for r in s2_results:
        d = r["date"]
        if d not in s2_dedup or r["roi_cloud_frac"] < s2_dedup[d]["roi_cloud_frac"]:
            s2_dedup[d] = r
    s2_results = [s2_dedup[d] for d in sorted(s2_dedup)]

    np.savez_compressed(
        s2_npz,
        mndwi=np.stack([r["mndwi"] for r in s2_results], axis=0),
        scl=np.stack([r["scl"] for r in s2_results], axis=0),
        utm_bbox=np.array(utm_bbox, dtype=np.float64),
        res_m=np.array([RES_M], dtype=np.float64),
    )
    pd.DataFrame(
        [{k: r[k] for k in ["date", "id", "scene_cloud_pct", "roi_cloud_frac", "valid_frac"]} for r in s2_results]
    ).to_parquet(CUBE_DIR / f"{key}_s2_meta.parquet", index=False)
    print(f"  [S2 Saved] {len(s2_results)} valid scenes -> {s2_npz.name}")

    # Re-sign S1 items cleanly before fetching
    catalog_s1 = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    s1_raw2 = list(catalog_s1.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=TIME_RANGE).items())
    s1_sel2 = _select_s1_items(s1_raw2, s2_sel_dates)

    s1_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(_fetch_one_s1, it, utm_bbox) for it in s1_sel2]
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                s1_results.append(res)
    s1_results.sort(key=lambda x: (x["date"], x["orbit_state"]))

    np.savez_compressed(
        s1_npz,
        vv_db=np.stack([r["vv_db"] for r in s1_results], axis=0),
        vh_db=np.stack([r["vh_db"] for r in s1_results], axis=0),
        utm_bbox=np.array(utm_bbox, dtype=np.float64),
        res_m=np.array([RES_M], dtype=np.float64),
    )
    pd.DataFrame(
        [{k: r[k] for k in ["date", "id", "orbit_state", "valid_frac"]} for r in s1_results]
    ).to_parquet(CUBE_DIR / f"{key}_s1_meta.parquet", index=False)

    dt = time.perf_counter() - t0
    print(f"  [S1 Saved] {len(s1_results)} valid scenes -> {s1_npz.name} (total reservoir time: {dt:.1f}s)")


def main() -> None:
    t_all = time.perf_counter()
    for key, info in RESERVOIRS.items():
        process_reservoir(key, info)
    print(f"\nALL 4 RESERVOIR DATACUBES COMPLETE in {(time.perf_counter() - t_all)/60:.1f} min.")


if __name__ == "__main__":
    main()

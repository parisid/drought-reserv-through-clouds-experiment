"""
Lightweight, zero-GDAL Cloud-Optimized GeoTIFF (COG) reader using HTTP Range
requests and pure-Python `tifffile`.

Reads ONLY the TIFF header and the exact internal tiles intersecting a target
UTM Zone 33N (EPSG:32633) bounding box — ideal for fast, reproducible satellite
workflows on standard CPU machines without GDAL/rasterio dependencies.
"""

from __future__ import annotations

import io
import math
import zlib
from typing import Tuple
import numpy as np
import requests
import tifffile


def _decode_tile_fallback(page: tifffile.TiffPage, raw: bytes, tidx: int) -> np.ndarray:
    """
    Decode a TIFF tile, with a pure-NumPy fallback for TIFF Floating-Point
    Predictor (Tag 317 == 3, Adobe Tech Note 3) with DEFLATE compression so we
    never require `imagecodecs` or GDAL native DLLs.
    """
    try:
        tile = page.decode(raw, tidx)[0]
        tile = np.squeeze(tile)
        if tile.ndim == 3:
            tile = tile[0]
        return tile
    except (ValueError, NotImplementedError) as exc:
        msg = str(exc)
        if "FLOATINGPOINT" in msg:
            decompressed = zlib.decompress(raw)
            th, tw = page.tilelength, page.tilewidth
            bps = page.dtype.itemsize  # 4 for float32, 8 for float64
            buf = np.frombuffer(decompressed, dtype=np.uint8).reshape(th, bps * tw)
            # Undo horizontal byte differencing (modulo 256)
            unpred = np.cumsum(buf, axis=1, dtype=np.uint8)
            # Undo byte-plane shuffling (MSB-first planes -> big-endian float)
            planes = unpred.reshape(th, bps, tw).transpose(0, 2, 1)
            be_dtype = f">f{bps}"
            tile = np.ascontiguousarray(planes).view(be_dtype).reshape(th, tw).astype(page.dtype)
            return tile
        if "packints_decode" in msg:
            decompressed = zlib.decompress(raw)
            th, tw = page.tilelength, page.tilewidth
            bits_per_sample = int(page.bitspersample)
            row_bytes = (tw * bits_per_sample + 7) // 8
            buf = np.frombuffer(decompressed, dtype=np.uint8).reshape(th, row_bytes)
            bits = np.unpackbits(buf, axis=1)[:, : tw * bits_per_sample].reshape(th, tw, bits_per_sample)
            weights = (1 << np.arange(bits_per_sample - 1, -1, -1, dtype=np.uint32))
            tile = (bits * weights).sum(axis=2, dtype=np.uint32).astype(page.dtype)
            if int(page.predictor) == 2:
                tile = np.cumsum(tile, axis=1, dtype=page.dtype)
            return tile
        raise


# ---------------------------------------------------------------------------
# Pure-Python WGS84 <-> UTM Zone 33N (EPSG:32633, central meridian 15E)
# Standard Karney/Snyder series expansion (< 1 mm accuracy across Sicily)
# ---------------------------------------------------------------------------
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2
_WGS84_EP2 = _WGS84_E2 / (1.0 - _WGS84_E2)
_UTM_K0 = 0.9996


def lonlat_to_utm33n(lon: float, lat: float) -> Tuple[float, float]:
    """Convert WGS84 (lon, lat) in degrees to UTM Zone 33N (easting, northing) in meters."""
    lon0_rad = math.radians(15.0)  # Zone 33 central meridian
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
    T = tan_lat ** 2
    C = _WGS84_EP2 * cos_lat ** 2
    A = cos_lat * (lon_rad - lon0_rad)

    e2 = _WGS84_E2
    M = _WGS84_A * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2 ** 3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = 500000.0 + _UTM_K0 * N * (
        A
        + (1 - T + C) * A ** 3 / 6.0
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * _WGS84_EP2) * A ** 5 / 120.0
    )
    northing = _UTM_K0 * (
        M
        + N * tan_lat * (
            A ** 2 / 2.0
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24.0
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * _WGS84_EP2) * A ** 6 / 720.0
        )
    )
    return easting, northing


def bbox_lonlat_to_utm33n(
    bbox: Tuple[float, float, float, float], snap_res: float = 20.0
) -> Tuple[float, float, float, float]:
    """Convert [min_lon, min_lat, max_lon, max_lat] to snapped UTM 33N [xmin, ymin, xmax, ymax]."""
    min_lon, min_lat, max_lon, max_lat = bbox
    corners = [
        lonlat_to_utm33n(min_lon, min_lat),
        lonlat_to_utm33n(min_lon, max_lat),
        lonlat_to_utm33n(max_lon, min_lat),
        lonlat_to_utm33n(max_lon, max_lat),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    xmin = math.floor(min(xs) / snap_res) * snap_res
    ymin = math.floor(min(ys) / snap_res) * snap_res
    xmax = math.ceil(max(xs) / snap_res) * snap_res
    ymax = math.ceil(max(ys) / snap_res) * snap_res
    return xmin, ymin, xmax, ymax


# ---------------------------------------------------------------------------
# Seekable HTTP Range File for tifffile
# ---------------------------------------------------------------------------
class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP supporting Range requests with a header cache."""

    def __init__(self, url: str, session: requests.Session | None = None, head_cache_bytes: int = 131072):
        self.url = url
        self.session = session or requests.Session()
        self._pos = 0

        # Fetch initial chunk (usually covers full COG IFD header + tile offset tables)
        resp = self.session.get(
            self.url,
            headers={"Range": f"bytes=0-{head_cache_bytes - 1}"},
            timeout=30,
        )
        resp.raise_for_status()
        self._head_cache = resp.content
        cr = resp.headers.get("Content-Range", "")
        if "/" in cr:
            self._size = int(cr.split("/")[-1])
        else:
            self._size = len(self._head_cache)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._pos + size - 1, self._size - 1)
        if end < self._pos:
            return b""

        # Serve from head cache if completely covered
        if end < len(self._head_cache):
            data = self._head_cache[self._pos : end + 1]
            self._pos = end + 1
            return data

        resp = self.session.get(
            self.url,
            headers={"Range": f"bytes={self._pos}-{end}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.content
        self._pos += len(data)
        return data

    def readinto(self, b: bytearray) -> int:
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


# ---------------------------------------------------------------------------
# Fast Tile-Cropped COG Reader aligned to a target UTM 33N grid
# ---------------------------------------------------------------------------
def _get_geotiff_transform(page: tifffile.TiffPage) -> Tuple[float, float, float, float]:
    """Return (origin_x, origin_y, pixel_size_x, pixel_size_y) for north-up GeoTIFF."""
    tags = page.tags
    if "ModelPixelScaleTag" in tags and "ModelTiepointTag" in tags:
        scale = tags["ModelPixelScaleTag"].value
        tie = tags["ModelTiepointTag"].value
        sx, sy = float(scale[0]), float(scale[1])
        # tie = (i, j, k, x, y, z)
        ox = float(tie[3]) - float(tie[0]) * sx
        oy = float(tie[4]) + float(tie[1]) * sy
        return ox, oy, sx, sy
    elif "ModelTransformationTag" in tags:
        m = tags["ModelTransformationTag"].value
        sx = float(m[0])
        sy = abs(float(m[5]))
        ox = float(m[3])
        oy = float(m[7])
        return ox, oy, sx, sy
    raise ValueError("GeoTIFF missing ModelPixelScaleTag/ModelTiepointTag and ModelTransformationTag.")


def read_cog_window_utm(
    url: str,
    utm_bbox: Tuple[float, float, float, float],
    target_res: float = 20.0,
    nearest: bool = False,
    session: requests.Session | None = None,
) -> np.ndarray:
    """
    Read a COG over HTTP (fetching only tiles intersecting `utm_bbox`) and
    resample onto the exact grid defined by `utm_bbox` at `target_res` meters.
    Returns a 2D numpy array of shape (H, W) with dtype float32 (or original int if nearest=True).
    """
    xmin, ymin, xmax, ymax = utm_bbox
    out_w = int(round((xmax - xmin) / target_res))
    out_h = int(round((ymax - ymin) / target_res))

    fh = HTTPRangeFile(url, session=session)
    with tifffile.TiffFile(fh) as tif:
        page = tif.pages[0]
        ox, oy, sx, sy = _get_geotiff_transform(page)
        img_h, img_w = page.shape[:2]

        # Pixel bounds in source image (with 1-pixel padding for safe interpolation)
        c0 = max(0, int(math.floor((xmin - ox) / sx)) - 1)
        c1 = min(img_w, int(math.ceil((xmax - ox) / sx)) + 2)
        r0 = max(0, int(math.floor((oy - ymax) / sy)) - 1)
        r1 = min(img_h, int(math.ceil((oy - ymin) / sy)) + 2)

        if c1 <= c0 or r1 <= r0:
            fill = 0 if nearest else np.nan
            return np.full((out_h, out_w), fill, dtype=page.dtype if nearest else np.float32)

        # Read only intersecting tiles (if tiled) or strips
        crop_h = r1 - r0
        crop_w = c1 - c0
        crop = np.zeros((crop_h, crop_w), dtype=page.dtype)

        if page.is_tiled:
            tw, th = page.tilewidth, page.tilelength
            tiles_across = math.ceil(img_w / tw)
            tr0, tr1 = r0 // th, (r1 - 1) // th
            tc0, tc1 = c0 // tw, (c1 - 1) // tw

            offsets = page.dataoffsets
            bytecounts = page.databytecounts

            for tr in range(tr0, tr1 + 1):
                for tc in range(tc0, tc1 + 1):
                    tidx = tr * tiles_across + tc
                    off = offsets[tidx]
                    cnt = bytecounts[tidx]
                    if cnt == 0:
                        continue
                    fh.seek(off)
                    raw = fh.read(cnt)
                    tile = _decode_tile_fallback(page, raw, tidx)

                    tile_r0 = tr * th
                    tile_c0 = tc * tw
                    # Overlap between tile and requested [r0:r1, c0:c1]
                    ov_r0 = max(r0, tile_r0)
                    ov_r1 = min(r1, tile_r0 + th)
                    ov_c0 = max(c0, tile_c0)
                    ov_c1 = min(c1, tile_c0 + tw)

                    crop[ov_r0 - r0 : ov_r1 - r0, ov_c0 - c0 : ov_c1 - c0] = tile[
                        ov_r0 - tile_r0 : ov_r1 - tile_r0,
                        ov_c0 - tile_c0 : ov_c1 - tile_c0,
                    ]
        else:
            full = page.asarray()
            crop = full[r0:r1, c0:c1]

    # Sample target pixel centers onto source crop coordinates
    target_xs = xmin + (np.arange(out_w) + 0.5) * target_res
    target_ys = ymax - (np.arange(out_h) + 0.5) * target_res

    src_cols = (target_xs - ox) / sx - 0.5 - c0
    src_rows = (oy - target_ys) / sy - 0.5 - r0

    if nearest:
        ri = np.clip(np.rint(src_rows).astype(int), 0, crop_h - 1)
        ci = np.clip(np.rint(src_cols).astype(int), 0, crop_w - 1)
        out = crop[np.ix_(ri, ci)]
        # Mask out-of-bounds pixels
        oob_r = (src_rows < -0.5) | (src_rows > crop_h - 0.5)
        oob_c = (src_cols < -0.5) | (src_cols > crop_w - 0.5)
        if np.any(oob_r) or np.any(oob_c):
            out = out.copy()
            out[oob_r, :] = 0
            out[:, oob_c] = 0
        return out
    else:
        from scipy.ndimage import map_coordinates

        rr, cc = np.meshgrid(src_rows, src_cols, indexing="ij")
        out = map_coordinates(
            crop.astype(np.float32),
            [rr, cc],
            order=1,
            mode="constant",
            cval=np.nan,
        ).astype(np.float32)
        return out

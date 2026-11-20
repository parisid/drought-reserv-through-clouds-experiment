# Towards Data Science — Submission Package & Checklist

## Article Metadata
- **Title**: `Tracking Drought Reservoirs Through Clouds With Sentinel-1 SAR and Python`
- **Subtitle**: `I tested optical satellite imagery, classic radar thresholding, and self-supervised LightGBM against 385 days of real dam volumes in Sicily — and a 1979 algorithm beat gradient boosting on three of four lakes`
- **Author**: Davide Parisi (`dp@davideparisi.com`, [davideparisi.com](https://www.davideparisi.com), ORCID: `0009-0002-9723-9685`)
- **Suggested Topics / Tags**: `Machine Learning`, `Geospatial`, `Data Science`, `Python`, `Remote Sensing`
- **Estimated Reading Time**: ~9–10 minutes (~2,050 words + 3 concise code blocks + 4 custom figures)

---

## Short Note for the TDS Submission Form ("What the piece covers & why it is timely")

```text
This hands-on technical article walks through building a reproducible, CPU-only (zero-GDAL) Python pipeline that fuses Sentinel-2 optical imagery with Sentinel-1 Synthetic Aperture Radar (SAR) to monitor drought-stressed water reservoirs through winter cloud cover. Rather than evaluating water masks against synthetic labels, I validate five approaches strictly out-of-time against 385 days of real official dam gauge readings (Jan 2025 – Feb 2026) across four Sicilian reservoirs recovering from severe drought.

Along with clean, adaptable Python code for HTTP-Range COG loading, cross-modal self-supervised LightGBM training (zero manual labels), and physically constrained hypsometric calibration, the piece highlights a counter-intuitive ML finding: per-scene 1979 Otsu thresholding outperformed global LightGBM on three sheltered valley reservoirs, while LightGBM cut error nearly in half on the widest, windiest lake. It also documents three concrete pitfalls (50/50 class-balancing area bias, free-intercept extrapolation failure, and EWMA smoothing lag during storm surges) that apply broadly to geospatial ML pipelines.
```

---

## Pre-Submission Editorial Checklist (TDS Guidelines)
- [x] **Short, specific Title + informative Subtitle** (no ALL-CAPS, no clickbait)
- [x] **Practitioner-first structure** (no academic paper numbering like RQ1–RQ5 or "Materials & Methods")
- [x] **Authentic human voice** grounded in real utility/water-sector experience
- [x] **100% real data** (1,880 real Sentinel-1/2 scenes + 385 days of official Sicilian dam gauge readings)
- [x] **Strong classical baseline + honest ML trade-offs** (Otsu vs. Pooled LightGBM vs. LORO Unseen-Lake LightGBM vs. Multi-Modal Fusion)
- [x] **Clean, concise code blocks** integrated directly into the narrative
- [x] **4 original high-DPI figures** (`Image by author`, no watermarks, no overlaid logos, crisp typography):
  1. `figures/fig1_cloud_blindness_and_radar_vision.png`
  2. `figures/fig2_daily_volume_tracking_4_reservoirs.png`
  3. `figures/fig3_method_comparison_4_reservoirs.png`
  4. `figures/fig4_hypsometry_and_sar_features.png`
- [x] **GitHub repo**: Linked in the article (`https://github.com/parisid/drought-reserv-through-clouds-experiment`) — remember to create & push the public repo before submitting!

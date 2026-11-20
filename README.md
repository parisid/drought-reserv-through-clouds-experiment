# Tracking Drought Reservoirs Through Clouds With Sentinel-1 SAR & Python

Reproducible, CPU-only, **zero-GDAL** Python pipeline fusing **Sentinel-2 L2A (optical)** and **Sentinel-1 RTC (SAR radar)** to monitor four drought-stressed Sicilian water reservoirs (**Lago Rosamarina**, **Lago Poma**, **Lago Castello**, and **Lago Fanaco**), validated strictly out-of-time against **385 days of official daily dam gauge readings (Jan 2025 – Feb 2026)**.

**Author:** Davide Parisi ([davideparisi.com](https://www.davideparisi.com) · [LinkedIn](https://www.linkedin.com/in/dparisi/) · [GitHub Repo](https://github.com/parisid/drought-reserv-through-clouds-experiment))

---

## Key Results (Out-of-Time Daily Validation, Jan 2025 – Feb 2026)

| Method | Mean Daily $R^2$ | Mean MAE (Mm³) | Mean RMSE (Mm³) | Mean NRMSE (% Cap.) | Freshness ($\le 7$d) | Worst Stale Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fusion (`S2 + S1-Otsu`)** | **0.837** | **1.44** | **1.91** | **4.08%** | **100.0%** | **6 days** |
| **Fusion (`S2 + S1-ML`)** | 0.783 | 1.49 | 2.16 | 5.20% | **100.0%** | **6 days** |
| **S1-Otsu (Classic SAR)** | 0.746 | 1.91 | 2.37 | 4.65% | 99.6% | 11 days |
| **S1-ML (Pooled LightGBM)** | 0.666 | 1.87 | 2.77 | 6.11% | 99.6% | 11 days |
| **S1-ML (LORO Unseen Lake)** | 0.616 | 1.93 | 2.97 | 6.30% | 99.6% | 11 days |
| **S2-Optical Only (`MNDWI`)** | 0.296 | 1.55 | 3.38 | 9.28% | 57.0% | **119 days** |

- **Sentinel-2 winter cloud blindness**: 69%–86% of winter passes obscured across the four reservoirs, with continuous blind streaks up to **75 days** on Lago Fanaco (`2025-12-12 -> 2026-02-25`).
- **Otsu vs. LightGBM**: Per-scene adaptive Otsu thresholding beats global LightGBM on the three narrow valley reservoirs (up to $R^2 = 0.936$ on Fanaco and $0.917$ on Castello), whereas LightGBM (using `VH` cross-polarization and local spatial texture) cuts SAR error nearly in half on the wide, windy **Lago Poma**.

## Repository Structure

```text
├── requirements.txt
├── src/
│   ├── cog_reader.py                 # Pure-Python HTTP Range COG reader (zero GDAL!)
│   ├── 01_fetch_and_inspect_dams.py  # Downloads official Sicilian reservoir CSVs
│   ├── 02_verify_stac_and_rois.py    # Planetary Computer STAC inventory & ROI sanity check
│   ├── 03_build_datacubes.py         # Parallel download of 1,880 S1/S2 scenes (2022–2026)
│   ├── 04_run_experiment.py          # Self-supervised LightGBM, LORO, Otsu, hypsometry & fusion
│   └── 05_generate_figures.py        # Generates the 4 publication-grade PNG figures
├── results/                          # Evaluation CSVs, feature importances & daily tracking tables
├── figures/                          # High-DPI PNG figures
├── TOWARDS_DATA_SCIENCE_ARTICLE.md   # Full article manuscript for Towards Data Science
└── TDS_SUBMISSION_NOTES.md           # Submission form pitch note & editorial checklist
```

## Quick Start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python src/01_fetch_and_inspect_dams.py
python src/03_build_datacubes.py
python src/04_run_experiment.py
python src/05_generate_figures.py
```

## Data Attribution
- **Reservoir Ground Truth**: Autorità di Bacino del Distretto Idrografico della Sicilia, structured by [Open Data Sicilia](https://github.com/opendatasicilia/emergenza-idrica-sicilia).
- **Satellite Imagery**: Copernicus Sentinel-1 RTC and Sentinel-2 L2A accessed via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).

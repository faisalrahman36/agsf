This is version 1 of Astronomy GMM Source Finder (AGSF).

Author: Syed Faisal ur Rahman

The core references used are :

Package: Scikit-learn
Algorithm: GMM (https://scikit-learn.org/stable/modules/mixture.html)


References:
1.  **Condon, J. J. (1997).** "Errors in Elliptical Gaussian Fits." *PASP*, 109, 166.
2.  **Dempster, A. P., et al. (1977).** "Maximum likelihood from incomplete data via the EM algorithm."
3.  **Schwarz, G. (1978).** "Estimating the dimension of a model." *The Annals of Statistics*.
4. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, F., Perrot, M., & Duchesnay, É. (2011).** Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
5. **Bradley, L., et al. (2025).** photutils: v2.3.0. Zenodo. https://doi.org/10.5281/zenodo.14856001

Acknowledgements:

The author acknowledges the assistance of multiple large language models (LLMs) in the engineering and technical implementation of AGSF. Specifically, the Gemini (Google) and Claude (Anthropic) models were employed as coding assistants for tasks including refining complex Python syntax, debugging the parallel processing logic in joblib, and optimizing array handling within the tiled FITS processing framework. The use of these models was limited to improving the efficiency of the implementation with explicit instructions from author especially to deal with the coding edge cases. All core theoretical framework, algorithmic contributions, astrophysical methodology (including the GMM fitting, deconvolution, and error derivation), validation tests, and scientific conclusions remain the responsibility of the author.

If using this then please cite:

**Rahman, S. F., Vardoulaki, E. (2025).** *AGSF: A  Probabilistic Source Finding Pipeline for Radio Interferometry using Gaussian Mixture Model (GMM). (In preparation)*.

Please add link to this repo [https://github.com/faisalrahman36/agsf] too in citation.


Please send feedback at: faisalrahman36@hotmail.com


# AGSF Source Finder - User Guide

**AGSF** (Astronomy GMM Source Finder) is a Python-based radio source finder that uses Gaussian Mixture Models to decompose complex astronomical sources. It is designed to match the sensitivity of industry standards (like PyBDSF) while providing robust de-blending for complex morphologies.

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install numpy scipy astropy photutils scikit-learn matplotlib joblib

```

### 2. Usage

Run the script from your terminal:

**Standard Run (uses defaults):**

```bash
python gmm_source_finder.py my_image.fits

```

**Run with Config (Recommended):**

```bash
python gmm_source_finder.py my_image.fits --config config.json

```

---

### 3. Configuration (`config.json`)

Create a `config.json` file with these settings. This **"Deep Field" configuration** is optimized for high sensitivity and multi-scale background estimation.

```json
{
    "output_dir": "gmm_results_optimized_v4",
    "save_plot": true,
    "mosaic": true,
    "n_jobs": -1,

    "detection_sigma": 3.0,
    "peak_snr_sigma": 5.0,
    "min_pix": 5,
    "box_sizes": [50, 100, 250],

    "multicomp_area_threshold": 2.0,
    "multicomp_snr_override": 15.0,
    "max_components": 6
}

```

#### 🔧 Key Parameters Explained

| Parameter | Recommended | Description & Use Case |
| --- | --- | --- |
| **`detection_sigma`** | **3.0** | **The "Wing" Detector.** Sets the island boundary low (3$\sigma$) to capture faint extended wings. **Use 3.0** for deep fields; **5.0** for bright-source-only catalogs. |
| **`peak_snr_sigma`** | **5.0** | **The Noise Filter.** Rejects noise ripples. A source is only kept if its *brightest pixel* > 5$\sigma$. **Increase to 5.5** if you see too many artifacts. |
| **`box_sizes`** | **`[50, 100, 250]`** | **Multi-Scale Background.** Estimates noise at multiple scales simultaneously. Critical for fields containing both compact sources and large diffuse galaxies. |
| **`multicomp_area_threshold`** | **2.0** | **Hybrid Logic.** Sources smaller than this (in beam areas) are forced to be **Single Gaussians**. Prevents overfitting small blobs. |
| **`multicomp_snr_override`** | **15.0** | **Bright Source Exception.** Allows small but bright (>15$\sigma$) sources to be split (e.g., tight double stars). |
| **`mosaic`** | `true` | **Memory Safe.** Splits large images into tiles. **Set to `true**` for images larger than 4k x 4k. |

---

### 4. Output Files

The pipeline generates these files in your output directory:

1. **`_components.csv`**: **The Science Catalog.**
* Contains positions (RA/DEC), Fluxes (Peak/Int), and deconvolved sizes for every fitted Gaussian. **Use this for analysis.**


2. **`_islands.csv`**: **The Detection Catalog.**
* Contains the raw properties of the detected "islands" (contours) before fitting. Useful for debugging dropouts.


3. **`diagnostic_plot.png`**: A visual overlay of detections on your FITS image.

### 5. Troubleshooting

* **Missing Faint Sources?** Lower `detection_sigma` to `3.0`.
* **Too Many Fake Sources?** Increase `peak_snr_sigma` to `5.5` or `6.0`.
* **Splitting Point Sources?** Increase `multicomp_area_threshold` to `2.5`.
* **Missed Diffuse Structure?** Add a larger box size, e.g., `[50, 100, 250, 400]`.

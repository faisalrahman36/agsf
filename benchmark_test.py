'''
AGSF vs PyBDSF Benchmark
Compares Component-to-Gaussian performance.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import bdsf
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import ZScaleInterval, ImageNormalize
import json

# Import GMM code
import gmm_source_finder_optimized_v7 as gmm_tool

# --- CONFIGURATION ---
#FITS_FILE = "cosmos144MHz_zoom.fits"
#FITS_FILE= "image_fullband.int.restored_3GHz.fits"
FITS_FILE= "sim_field.fits"
#OUTPUT_DIR = "benchmark_Eleni3GHz_gmmv7"
OUTPUT_DIR = "benchmark_sim_field_gmm7_conf5"
CONFIG_FILE = "config5.json"
print(FITS_FILE, CONFIG_FILE, OUTPUT_DIR)
print("abc123")

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# --- 1. RUN PYBDSF (GAUSSIANS) ---
def run_pybdsf_gaussians(fits_path):
    print(f"\n>>> RUNNING PYBDSF on {fits_path}")
    try:
        # Match thresholds to GMM config
        img = bdsf.process_image(
            fits_path,
            thresh_pix=5.0, # peak_snr_sigma in GMM
            thresh_isl=3.0, # detection_sigma in GMM
            adaptive_thresh=True,
            rms_box=(150, 50),
            atrous_do=False,
            format='csv',
            clobber=True,
            quiet=True
        )
        
        # Export 'gaul' (Gaussians) for GMM components comparison
        gau_file = os.path.join(OUTPUT_DIR, "pybdsf_gaussians.csv")
        img.write_catalog(outfile=gau_file, catalog_type='gaul', format='csv', clobber=True)
        
        #sources or srl for GMM islands comparison
        srl_file = os.path.join(OUTPUT_DIR, "pybdsf_sources.csv")

        img.write_catalog(outfile=srl_file, format='csv', catalog_type='srl')
        
        
        print(f"PyBDSF Found: {img.nsrc} Sources, {len(img.gaussians)} Gaussians")
        return gau_file
    except Exception as e:
        print(f"Error running PyBDSF: {e}")
        return None

# --- 2. RUN GMM-RADIO ---
def run_gmm_full(fits_path):
    print(f"\n>>> RUNNING GMM-RADIO on {fits_path}")
    
    cfg = gmm_tool.DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg.update(json.load(f))
    
    cfg['output_dir'] = OUTPUT_DIR
    cfg['save_plot'] = False 
    
    # Run the GMM pipeline
    
    hdul = fits.open(fits_path)
    header = hdul[0].header
    data = np.squeeze(hdul[0].data)
    wcs = WCS(header).celestial
    if data.ndim > 2: data = data[0] if data.ndim == 3 else data[0,0]
    
    # Pixel Scale
    pix_scales = proj_plane_pixel_scales(wcs)
    pscale = pix_scales[0] # Degrees
    
    beam = gmm_tool.get_beam_info(header, pscale)
    
    f_isl = os.path.join(OUTPUT_DIR, "gmm_islands.csv")
    f_comp = os.path.join(OUTPUT_DIR, "gmm_components.csv")
    
    if cfg.get('mosaic', True):
        gmm_tool.run_mosaic(data, wcs, beam, pscale, cfg, f_isl, f_comp, OUTPUT_DIR)
    else:
        cands, _ = gmm_tool.detect_on_data(data, wcs, cfg)
        gmm_tool.process_candidates(cands, beam, pscale, cfg, f_isl, f_comp, "FullFrame")
    return f_comp

# --- 3. COMPARISON & PLOTTING ---
def compare_catalogs(gmm_path, pyb_path, fits_path):
    print(f"\n--- Comparing Catalogs ---")
    gmm = pd.read_csv(gmm_path)
    
    # PyBDSF Reader (Handles comments/headers)
    try:
        # Find header line starting with "Source_id" or "Gaus_id"
        header_row = 0
        with open(pyb_path, 'r') as f:
            for i, line in enumerate(f):
                if "RA" in line and "DEC" in line:
                    header_row = i
                    break
        pyb = pd.read_csv(pyb_path, skiprows=header_row, skipinitialspace=True)
        pyb.columns = [c.strip().replace('#', '') for c in pyb.columns] # Clean columns
    except:
        print("Failed to read PyBDSF catalog.")
        return

    print(f"GMM Components:    {len(gmm)}")
    print(f"PyBDSF Gaussians:  {len(pyb)}")
    
    # Match Catalogues
    c_gmm = SkyCoord(ra=gmm['RA'].values*u.deg, dec=gmm['DEC'].values*u.deg)
    c_pyb = SkyCoord(ra=pyb['RA'].values*u.deg, dec=pyb['DEC'].values*u.deg)
    
    idx, d2d, _ = c_gmm.match_to_catalog_sky(c_pyb)
    match_mask = d2d < 1.0 * u.arcsec # Tight matching for components
    
    matches = gmm[match_mask].copy()
    matches['Ref_Flux'] = pyb.iloc[idx[match_mask]]['Total_flux'].values
    matches['Ref_Peak'] = pyb.iloc[idx[match_mask]]['Peak_flux'].values
    
    print(f"Matched Components: {len(matches)}")
    
    # Plot 1: Flux Comparison
    plt.figure(figsize=(6, 6))
    x = matches['Ref_Flux']
    y = matches['Total_flux']
    plt.scatter(x, y, alpha=0.5, s=10, c='blue', label='Total Flux')
    plt.plot([min(x), max(x)], [min(x), max(x)], 'r--')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('PyBDSF Flux (Jy)'); plt.ylabel('GMM Flux (Jy)')
    plt.title('Flux Recovery (Gaussian to Gaussian)')
    plt.savefig(os.path.join(OUTPUT_DIR, "flux_comparison.png"))
    plt.close()

    # Plot 2: Overlay
    plot_overlay(gmm, pyb, fits_path)

def plot_overlay(gmm, pyb, fits_path):
    print("Generating Overlay Map...")
    with fits.open(fits_path) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
        wcs = WCS(header).celestial
        if data.ndim > 2: data = data[0] if data.ndim == 3 else data[0,0]

    # scale factor
    pix_scales = proj_plane_pixel_scales(wcs)
    deg_per_pix = pix_scales[0]

    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(data, interval=ZScaleInterval())
    ax.imshow(data, origin='lower', cmap='Greys', norm=norm)
    
    # Plot PyBDSF (Lime)
    # PyBDSF 'Maj'/'Min' are typically FWHM in DEGREES
    for _, row in pyb.iterrows():
        try:
            x, y = wcs.world_to_pixel_values(row['RA'], row['DEC'])
            maj_pix = row['Maj'] / deg_per_pix
            min_pix = row['Min'] / deg_per_pix
            e = Ellipse((x, y), width=min_pix, height=maj_pix, angle=row['PA']+90,
                        edgecolor='lime', facecolor='none', lw=1.0)
            ax.add_patch(e)
        except: pass

    # Plot GMM (Red)
    # GMM 'Maj'/'Min' are FWHM in ARCSEC
    for _, row in gmm.iterrows():
        try:
            x, y = wcs.world_to_pixel_values(row['RA'], row['DEC'])
            maj_pix = (row['Maj'] / 3600.0) / deg_per_pix
            min_pix = (row['Min'] / 3600.0) / deg_per_pix
            e = Ellipse((x, y), width=min_pix, height=maj_pix, angle=row['PA']+90,
                        edgecolor='red', facecolor='none', lw=1.0, linestyle='--')
            ax.add_patch(e)
        except: pass

    # Legend
    from matplotlib.lines import Line2D
    lines = [Line2D([0],[0], color='lime', lw=2), Line2D([0],[0], color='red', lw=2, ls='--')]
    ax.legend(lines, ['PyBDSF (Gaussians)', 'GMM (Components)'])
    
    plt.savefig(os.path.join(OUTPUT_DIR, "overlay_map.png"), dpi=200)
    plt.close()

if __name__ == "__main__":
    setup_dirs()
    pyb_gau = run_pybdsf_gaussians(FITS_FILE)
    gmm_comp = run_gmm_full(FITS_FILE)
    if pyb_gau and gmm_comp:
        compare_catalogs(gmm_comp, pyb_gau, FITS_FILE)
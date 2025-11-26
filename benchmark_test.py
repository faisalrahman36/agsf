import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import bdsf
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.visualization import ZScaleInterval, ImageNormalize
import json

# Import your GMM code
import gmm_source_finder
# --- CONFIGURATION ---
FITS_FILE = "cosmos144MHz_zoom.fits"
OUTPUT_DIR = "benchmark_full"
PYBDSF_THRESH_PIX = 5.0
PYBDSF_THRESH_ISL = 3.0
CONFIG_FILE = "config.json"

def setup_dirs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# --- 1. RUN PYBDSF (Sources Only) ---
def run_pybdsf_full(fits_path):
    print("\n" + "="*40)
    print(">>> RUNNING PYBDSF")
    print("="*40)
    try:
        img = bdsf.process_image(
            fits_path,
            thresh_pix=PYBDSF_THRESH_PIX,
            thresh_isl=PYBDSF_THRESH_ISL,
            adaptive_thresh=True,
            rms_box=(150, 50),
            atrous_do=False,
            format='csv',
            clobber=True,
            quiet=True
        )
        
        # Export ONLY Source List (srl) - safest option
        srl_file = os.path.join(OUTPUT_DIR, "pybdsf_sources.csv")
        img.write_catalog(outfile=srl_file, catalog_type='srl', format='csv', clobber=True)
        
        print(f"PyBDSF Found: {img.nsrc} Sources")
        return srl_file
    except Exception as e:
        print(f"Error running PyBDSF: {e}")
        return None

# --- 2. RUN GMM-RADIO (Islands + Components) ---
def run_gmm_full(fits_path):
    print("\n" + "="*40)
    print(">>> RUNNING GMM-RADIO")
    print("="*40)
    
    cfg = gmm_source_finder.DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg.update(json.load(f))
    
    cfg['output_dir'] = OUTPUT_DIR
    cfg['save_plot'] = False 
    
    hdul = fits.open(fits_path)
    header = hdul[0].header
    data = np.squeeze(hdul[0].data)
    wcs = WCS(header).celestial
    if data.ndim > 2: data = data[0] if data.ndim == 3 else data[0,0]
    
    try: pscale = abs(wcs.wcs.cdelt[1])
    except: pscale = 1.0/3600.0
    beam = gmm_source_finder.get_beam_info(header, pscale)
    
    f_isl = os.path.join(OUTPUT_DIR, "gmm_islands.csv")
    f_comp = os.path.join(OUTPUT_DIR, "gmm_components.csv")
    
    if cfg['mosaic']:
        gmm_source_finder.run_mosaic(data, wcs, beam, pscale, cfg, f_isl, f_comp, OUTPUT_DIR)
    else:
        gmm_source_finder.run_standard(data, wcs, beam, pscale, cfg, f_isl, f_comp, OUTPUT_DIR)
    
    return f_comp

# --- ROBUST READER ---
def read_catalog_smart(filename):
    """
    Reads PyBDSF catalogs handling variable headers and comments.
    """
    try:
        # 1. Find Header Line
        header_row = 0
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Check for common PyBDSF column names
            # Note: PyBDSF sometimes puts units in header like "RA (deg)"
            if "Source_id" in line or "RA" in line:
                header_row = i
                break
        
        # 2. Read
        # We treat '#' as a comment to avoid it being part of column names
        # But we need to make sure we don't skip the header row if it starts with #
        
        # Try reading with 'header=header_row'
        df = pd.read_csv(filename, header=header_row, skipinitialspace=True)
        
        # Clean up column names (remove # and spaces)
        df.columns = [c.replace('#', '').strip() for c in df.columns]
        
        print(f"DEBUG: Columns found in {filename}: {list(df.columns)}")

        # 3. Normalize Columns
        col_map = {}
        for col in df.columns:
            c = col.upper()
            # General Mappings
            if 'RA' in c and 'ERR' not in c: col_map[col] = 'RA'
            if 'DEC' in c and 'ERR' not in c: col_map[col] = 'DEC'
            if 'TOTAL_FLUX' in c and 'ERR' not in c: col_map[col] = 'Total_flux'
            if 'MAJ' in c and 'ERR' not in c: col_map[col] = 'Maj'
            if 'MIN' in c and 'ERR' not in c: col_map[col] = 'Min'
            if 'PA' in c and 'ERR' not in c: col_map[col] = 'PA'
            
        df.rename(columns=col_map, inplace=True)
        
        # Verify
        if 'RA' not in df.columns:
            print("CRITICAL ERROR: 'RA' column not found after mapping.")
            # Fallback: Try to assume column positions for standard PyBDSF SRL
            # Source_id, Isl_id, RA, E_RA, DEC, E_DEC, Total_flux, ...
            if len(df.columns) >= 7:
                print("Attempting positional mapping...")
                df.rename(columns={df.columns[2]: 'RA', df.columns[4]: 'DEC', df.columns[6]: 'Total_flux'}, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error reading catalog {filename}: {e}")
        return pd.DataFrame()

# --- COMPARISON LOGIC ---
def compare_catalogs(gmm_path, pyb_path, fits_path):
    print(f"\n--- Comparing Components ---")
    gmm = pd.read_csv(gmm_path)
    pyb = read_catalog_smart(pyb_path)
    
    if pyb.empty or gmm.empty:
        print(f"Skipping Comparison (Empty/Missing File)")
        return

    print(f"GMM Count:    {len(gmm)}")
    print(f"PyBDSF Count: {len(pyb)}")
    
    # Check if 'RA' exists now
    if 'RA' not in pyb.columns:
        print("Error: Could not identify RA column in PyBDSF file. Aborting match.")
        return

    # Matching
    c_gmm = SkyCoord(ra=gmm['RA'].values*u.deg, dec=gmm['DEC'].values*u.deg)
    c_pyb = SkyCoord(ra=pyb['RA'].values*u.deg, dec=pyb['DEC'].values*u.deg)
    
    idx, d2d, _ = c_gmm.match_to_catalog_sky(c_pyb)
    match_mask = d2d < 2.0 * u.arcsec # Match radius
    
    matches = gmm[match_mask].copy()
    matches['Ref_Flux'] = pyb.iloc[idx[match_mask]]['Total_flux'].values
    
    print(f"Matched:      {len(matches)}")
    
    # Plot Flux
    plt.figure(figsize=(6, 6))
    
    # Scale to mJy if appropriate
    scale = 1000.0
    unit = "mJy"
    if matches['Total_flux'].max() > 1.0: # If > 1 Jy, keep in Jy
        scale = 1.0
        unit = "Jy"

    x = matches['Ref_Flux'] * scale
    y = matches['Total_flux'] * scale
    
    plt.scatter(x, y, alpha=0.6, c='blue')
    if len(x) > 0:
        mx = max(x.max(), y.max())
        mn = min(x.min(), y.min())
        plt.plot([mn, mx], [mn, mx], 'r--')
        
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel(f'PyBDSF Flux ({unit})'); plt.ylabel(f'GMM Flux ({unit})')
    plt.title(f'Flux Comparison')
    plt.savefig(os.path.join(OUTPUT_DIR, f"compare_flux.png"))
    print(f"Saved plot: compare_flux.png")
    
    # Generate Overlay
    plot_final_overlay(gmm, pyb, fits_path)

def plot_final_overlay(gmm_comp, pyb_comp, fits_path):
    print("\n--- Generating Master Overlay ---")
    with fits.open(fits_path) as hdul:
        data = np.squeeze(hdul[0].data)
        header = hdul[0].header
        wcs = WCS(header).celestial
        if data.ndim > 2: data = data[0] if data.ndim == 3 else data[0,0]

    fig = plt.figure(figsize=(14, 14))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(data, interval=ZScaleInterval())
    ax.imshow(data, origin='lower', cmap='Greys', norm=norm)
    
    # Helper to plot ellipse
    def plot_cat(df, color, style, label, scale_factor=1.0):
        for _, row in df.iterrows():
            try:
                x, y = wcs.world_to_pixel_values(row['RA'], row['DEC'])
                maj = row['Maj'] * scale_factor
                min = row['Min'] * scale_factor
                e = Ellipse((x, y), width=min, height=maj, angle=row['PA']+90,
                            edgecolor=color, facecolor='none', lw=1.5, linestyle=style)
                ax.add_patch(e)
            except: pass
    
    # Pixel Scale for conversion
    try:
        pix_scale = abs(header['CDELT2'])
    except:
        pix_scale = 1.0/3600.0
    
    # 1. PyBDSF Components (Green)
    # PyBDSF Maj/Min are usually in DEGREES. We need pixels.
    # Pixels = Degrees / PixelScale
    if 'Maj' in pyb_comp.columns:
        plot_cat(pyb_comp, 'lime', '-', 'PyBDSF', scale_factor=1.0/pix_scale)
    
    # 2. GMM Components (Red)
    # GMM Maj/Min are in ARCSEC. We need pixels.
    # Pixels = (Arcsec / 3600) / PixelScale
    if 'Maj' in gmm_comp.columns:
        plot_cat(gmm_comp, 'red', '--', 'GMM', scale_factor=(1.0/3600.0)/pix_scale)

    from matplotlib.lines import Line2D
    cl = [Line2D([0],[0], color='lime', lw=2), Line2D([0],[0], color='red', lw=2, ls='--')]
    ax.legend(cl, ['PyBDSF', 'GMM-Radio'])
    
    plt.savefig(os.path.join(OUTPUT_DIR, "master_overlay.png"), dpi=200)
    print("Saved master_overlay.png")

# --- MAIN DRIVER ---
if __name__ == "__main__":
    setup_dirs()
    
    # 1. Run Tools
    pyb_srl = run_pybdsf_full(FITS_FILE)
    gmm_comp = run_gmm_full(FITS_FILE)
    
    # 2. Compare
    if pyb_srl and gmm_comp:
        compare_catalogs(gmm_comp, pyb_srl, FITS_FILE)
'''
Astronomy GMM Source Finder (AGSF)
Author: Syed Faisal ur Rahman


'''

import argparse
import numpy as np
import warnings
import csv
import sys
import gc
import json
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Core module
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed

# To check if diagnostics file is there
try:
    from gmm_diagnostics import DiagnosticsRunner
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

warnings.filterwarnings('ignore')

# DEFAULTS 
DEFAULT_CONFIG = {
    "output_dir": "gmm_results",
    "save_plot": True,
    "mosaic": True,
    "tile_size": 2500,
    "padding": 100,
    "box_sizes": [100],
    "detection_sigma": 5.0,
    "min_pix": 5,
    "n_jobs": -1,
    "exclusion_radius": 5.0,
    "max_components": 6,
    
    # Hybrid De-blending & Dynamic Logic 
    "multicomp_area_threshold": 2.0,
    "multicomp_snr_override": 15.0,
    "resample_factor": 100,
    "min_samples": 500,
    "max_samples": 10000
}

# SYSTEM SETUP 
def setup_environment(config_path, cli_no_mosaic):
    cfg = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        print(f"Loading Config: {config_path}")
        with open(config_path, 'r') as f:
            cfg.update(json.load(f))
            
    if cli_no_mosaic:
        cfg['mosaic'] = False
        print(">> CLI Override: Mosaic Mode DISABLED")

    if "box_size" in cfg and "box_sizes" not in cfg:
        cfg["box_sizes"] = [cfg["box_size"]]
        
    out_dir = cfg.get('output_dir', 'gmm_output')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    return cfg, out_dir

# --- UTILS ---
def get_beam_info(header, pixel_scale):
    try:
        bmaj = header['BMAJ']
        bmin = header['BMIN']
        bpa = header.get('BPA', 0.0)
        theta_maj = bmaj / pixel_scale
        theta_min = bmin / pixel_scale
        beam_area = (np.pi * theta_maj * theta_min) / (4 * np.log(2))
        return bmaj, bmin, bpa, beam_area
    except KeyError:
        return 1.0/3600, 1.0/3600, 0.0, 1.0

def deconvolve(maj, min, pa, bmaj, bmin, bpa):
    if maj < bmaj: return 0.0, 0.0, 0.0
    dc_maj = np.sqrt(max(0, maj**2 - bmaj**2))
    dc_min = np.sqrt(max(0, min**2 - bmin**2))
    return dc_maj, dc_min, pa

def calculate_errors(flux_peak, flux_int, maj, min, local_rms, snr):
    if snr <= 0: return 0, 0, 0, 0, 0, 0, 0
    snr = max(snr, 0.1)
    err_peak = flux_peak * np.sqrt((1/snr)**2 + 0.01**2)
    err_int = flux_int * np.sqrt((1/snr)**2 + 0.01**2)
    err_maj = maj / snr
    err_min = min / snr
    err_pa = 10.0 / snr
    err_ra = maj / (2*snr)
    err_dec = min / (2*snr)
    return err_peak, err_int, err_maj, err_min, err_pa, err_ra, err_dec

# GMM FITTER 
def fit_island_worker(task):
    island_id = task['id']
    cutout = task['cutout']
    mask = task['mask']
    wcs_slice = task['wcs']
    beam = task['beam']
    pix_scale = task['pix_scale']
    config = task['config']
    box_origin = task.get('box_origin', 0)

    # Use passed RMS 
    rms = task['rms']
    if rms == 0 or np.isnan(rms): return []

    valid = (mask) & (~np.isnan(cutout))
    y, x = np.indices(cutout.shape)
    flux_vals = cutout[valid]
    
    if len(flux_vals) < 5: return []

    X = np.vstack([x[valid], y[valid]]).T
    
    # Dynamic Sampling
    island_peak = np.max(flux_vals)
    island_snr = island_peak / rms
    n_dynamic = int(np.clip(island_snr * config.get('resample_factor', 100), 
                            config.get('min_samples', 500), 
                            config.get('max_samples', 10000)))
    n_dynamic = min(n_dynamic, len(flux_vals) * 10)

    weights = np.maximum(flux_vals, 0)
    if np.sum(weights) == 0: return []
    weights /= np.sum(weights)
    
    rng = np.random.default_rng(42)
    X_samp = X[rng.choice(len(X), size=n_dynamic, p=weights)]
    
    bmaj, bmin, bpa, beam_area = beam
    min_variance = 0.1 * beam_area 

    # Hybrid Logic
    limit_threshold = config.get('multicomp_area_threshold', 2.0) * beam_area
    snr_override = config.get('multicomp_snr_override', 15.0)

    if len(flux_vals) >= limit_threshold or island_snr >= snr_override:
        max_comp = config['max_components']
    else:
        max_comp = 1

    best_bic = np.inf
    best_gmm = None
    
    for n in range(1, max_comp + 1):
        try:
            gmm = GaussianMixture(n_components=n, 
                                  covariance_type='full', 
                                  reg_covar=min_variance,
                                  random_state=42)
            gmm.fit(X_samp)
            current_bic = gmm.bic(X_samp)
            if current_bic < best_bic:
                best_bic = current_bic
                best_gmm = gmm
        except: continue
            
    if not best_gmm: return []

    comps = []
    
    for i in range(best_gmm.n_components):
        mx, my = best_gmm.means_[i]
        vals, vecs = np.linalg.eigh(best_gmm.covariances_[i])
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        
        dy, dx = vecs[:, 0]
        pa = np.degrees(np.arctan2(dx, dy)) % 180
        
        maj_pix = np.sqrt(vals[0]) * 2.355
        min_pix = np.sqrt(vals[1]) * 2.355
        maj_deg = maj_pix * pix_scale
        min_deg = min_pix * pix_scale
        
        comp_weight = best_gmm.weights_[i]
        raw_int_flux_jy = (np.sum(flux_vals) / beam_area) * comp_weight
        
        gaussian_area_pix = 1.133 * maj_pix * min_pix
        peak_flux = raw_int_flux_jy / (gaussian_area_pix / beam_area)
        
         
        if peak_flux < 1.0 * rms:
            continue

        snr = peak_flux / rms
        dc_maj, dc_min, dc_pa = deconvolve(maj_deg, min_deg, pa, bmaj, bmin, bpa)
        errs = calculate_errors(peak_flux, raw_int_flux_jy, maj_deg, min_deg, rms, snr)

        try:
            sky = wcs_slice.pixel_to_world(mx, my)
            ra_val, dec_val = sky.ra.deg, sky.dec.deg
        except:
            ra_val, dec_val = 0.0, 0.0

        comps.append({
            'Island_id': island_id,
            'RA': ra_val, 'E_RA': errs[5],
            'DEC': dec_val, 'E_DEC': errs[6],
            'Total_flux': raw_int_flux_jy, 'E_Total_flux': errs[1],
            'Peak_flux': peak_flux, 'E_Peak_flux': errs[0],
            'Maj': maj_deg*3600, 'E_Maj': errs[2]*3600,
            'Min': min_deg*3600, 'E_Min': errs[3]*3600,
            'PA': pa, 'E_PA': errs[4],
            'DC_Maj': dc_maj*3600, 'DC_Min': dc_min*3600, 'DC_PA': dc_pa,
            'RMS': rms, 'S_Code': 'S' if best_gmm.n_components==1 else 'C',
            'Detection_Box': box_origin
        })
    return comps

def detect_on_data(data, wcs, config, edge_info=None):
    all_candidates = []
    h, w = data.shape
    
    for box in config['box_sizes']:
        try:
            bkg = Background2D(data, (box, box), filter_size=(3, 3),
                               sigma_clip=SigmaClip(sigma=3.0), 
                               bkg_estimator=MedianBackground())
            sub = data - bkg.background
            rms = bkg.background_rms if hasattr(bkg, 'background_rms') else bkg.rms
            
            thresh = config['detection_sigma'] * rms
            segm = detect_sources(sub, thresh, npixels=config['min_pix'])
            if segm is None: continue
            
            cat = SourceCatalog(sub, segm, error=rms, wcs=wcs)
            
            for source in cat:
                cx, cy = source.centroid
                if edge_info:
                    is_left, is_right, is_bottom, is_top = edge_info
                    pad = config['padding']
                    if (not is_left) and (cx < pad): continue
                    if (not is_right) and (cx > w - pad): continue
                    if (not is_bottom) and (cy < pad): continue
                    if (not is_top) and (cy > h - pad): continue

                sl = source.slices
                
                try:
                    maj_sigma = getattr(source.semimajor_sigma, 'value', source.semimajor_sigma)
                    min_sigma = getattr(source.semiminor_sigma, 'value', source.semiminor_sigma)
                    orientation = getattr(source.orientation, 'value', source.orientation)
                    peak_val = source.max_value
                except:
                    maj_sigma, min_sigma, orientation, peak_val = np.nan, np.nan, np.nan, 0

                cand = {
                    'cx': cx, 'cy': cy, 'flux': source.segment_flux,
                    'peak': peak_val,
                    'maj_sig': maj_sigma, 'min_sig': min_sigma, 'orient': orientation,
                    'cutout': sub[sl].copy(), 'mask': (segm.data[sl] == source.label),
                    'rms': rms[int(cy), int(cx)], 'wcs': wcs.slice(sl),
                    'box': box
                }
                all_candidates.append(cand)
        except: continue

    if not all_candidates: return [], None
    
    all_candidates.sort(key=lambda x: x['flux'], reverse=True)
    unique = []
    exclusion_sq = config['exclusion_radius']**2

    while all_candidates:
        curr = all_candidates.pop(0)
        unique.append(curr)
        all_candidates = [c for c in all_candidates 
                          if ((c['cx']-curr['cx'])**2 + (c['cy']-curr['cy'])**2) > exclusion_sq]
    
    return unique, segm

def run_standard(data, wcs, beam, scale, config, f_isl, f_comp, out_dir):
    print("--- Mode: Standard (Full Frame) ---")
    cands, segm = detect_on_data(data, wcs, config)
    comps = process_candidates(cands, beam, scale, config, f_isl, f_comp, "FullFrame")
    if config.get('save_plot', False) and comps:
        plot_path = os.path.join(out_dir, "diagnostic_plot.png")
        generate_production_plot(data, wcs, segm, comps, plot_path, scale)

def run_mosaic(data, full_wcs, beam, scale, config, f_isl, f_comp, out_dir):
    print("--- Mode: Mosaic (Tiled) ---")
    with open(f_isl, 'w') as f: pass 
    with open(f_comp, 'w') as f: pass
    
    ts = config['tile_size']
    pad = config['padding']
    h, w = data.shape
    all_components_for_plot = []
    
    for y in range(0, h, ts):
        for x in range(0, w, ts):
            x0, x1 = max(0, x-pad), min(w, x+ts+pad)
            y0, y1 = max(0, y-pad), min(h, y+ts+pad)
            edge_info = (x0==0, x1==w, y0==0, y1==h)
            tile_id = f"T{x}_{y}"
            
            print(f"Processing {tile_id}...", end='\r')
            tile_data = data[y0:y1, x0:x1]
            tile_wcs = full_wcs[y0:y1, x0:x1]
            cands, _ = detect_on_data(tile_data, tile_wcs, config, edge_info)
            tile_comps = process_candidates(cands, beam, scale, config, f_isl, f_comp, tile_id, append=True)
            if tile_comps: all_components_for_plot.extend(tile_comps)
            del tile_data
            gc.collect()

    print(f"\nMosaic Complete.")
    if config.get('save_plot', False) and all_components_for_plot:
        if data.size < 8000**2:
            plot_path = os.path.join(out_dir, "mosaic_plot.png")
            generate_production_plot(data, full_wcs, None, all_components_for_plot, plot_path, scale)

def process_candidates(cands, beam, scale, config, f_isl, f_comp, group_id, append=False):
    if not cands: return []
    bmaj, bmin, bpa, beam_area = beam
    island_list = []
    fit_tasks = []
    
    for i, cand in enumerate(cands):
        uid = f"{group_id}_{i+1}"
        sky = cand['wcs'].pixel_to_world(cand['cutout'].shape[1]//2, cand['cutout'].shape[0]//2)
        
        maj_deg = (cand['maj_sig'] * 2.355) * scale if not np.isnan(cand['maj_sig']) else 0
        min_deg = (cand['min_sig'] * 2.355) * scale if not np.isnan(cand['min_sig']) else 0
        pa = (90 - np.degrees(cand['orient'])) % 180 if not np.isnan(cand['orient']) else 0.0

        total_flux_jy = cand['flux'] / beam_area
        peak_flux_jy = cand['peak']
        local_rms = cand['rms']
        
        isl_snr = peak_flux_jy / local_rms
        errs = calculate_errors(peak_flux_jy, total_flux_jy, maj_deg, min_deg, local_rms, isl_snr)

        island_list.append({
            'Island_id': uid, 
            'RA': sky.ra.deg, 'E_RA': errs[5],
            'DEC': sky.dec.deg, 'E_DEC': errs[6],
            'Total_flux': total_flux_jy, 'E_Total_flux': errs[1],
            'Peak_flux': peak_flux_jy, 'E_Peak_flux': errs[0],
            'Maj': maj_deg*3600, 'E_Maj': errs[2]*3600,
            'Min': min_deg*3600, 'E_Min': errs[3]*3600,
            'PA': pa, 'E_PA': errs[4],
            'Isl_rms': local_rms,
            'BPA': bpa,
            'Detection_Box': cand['box']
        })
        
        fit_tasks.append({
            'id': uid, 'cutout': cand['cutout'], 'mask': cand['mask'],
            'rms': cand['rms'], 'wcs': cand['wcs'], 'beam': beam,
            'pix_scale': scale, 'config': config, 'box_origin': cand['box']
        })

    results = Parallel(n_jobs=config['n_jobs'])(delayed(fit_island_worker)(t) for t in fit_tasks)
    comps = [c for sub in results for c in sub]

    mode = 'a' if append else 'w'
    if island_list:
        write_head = (not append) or (not os.path.exists(f_isl)) or (os.path.getsize(f_isl) == 0)
        with open(f_isl, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=island_list[0].keys())
            if write_head: writer.writeheader()
            writer.writerows(island_list)
            
    if comps:
        write_head = (not append) or (not os.path.exists(f_comp)) or (os.path.getsize(f_comp) == 0)
        with open(f_comp, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=comps[0].keys())
            if write_head: writer.writeheader()
            writer.writerows(comps)
    return comps

def generate_production_plot(data, wcs, segm, components, output_path, pixel_scale):
    print(f"Generating Visualization: {output_path} ...")
    fig = plt.figure(figsize=(14, 12))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(data, interval=ZScaleInterval())
    im = ax.imshow(data, origin='lower', cmap='Greys', norm=norm)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Flux Density (Jy/beam)', rotation=270, labelpad=20)

    if segm is not None and data.size < 5000**2: 
        mask = (segm.data > 0).astype(float)
        ax.contour(mask, levels=[0.5], colors='cyan', linewidths=0.7, alpha=0.9)

    for c in components:
        sky = SkyCoord(c['RA'], c['DEC'], unit='deg')
        x_pix, y_pix = wcs.world_to_pixel(sky)
        maj_pix = (c['Maj'] / 3600.0) / pixel_scale
        min_pix = (c['Min'] / 3600.0) / pixel_scale
        plot_angle = 90 + c['PA'] 
        e = Ellipse((x_pix, y_pix), width=min_pix, height=maj_pix, angle=plot_angle,
                    edgecolor='red', facecolor='none', lw=1.2, linestyle='-')
        ax.add_patch(e)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fits_file")
    parser.add_argument("--prefix", default="gmm_run")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--no-mosaic", action="store_true")
    args = parser.parse_args()
    
    cfg, out_dir = setup_environment(args.config, args.no_mosaic)
    hdul = fits.open(args.fits_file, memmap=True)
    header = hdul[0].header
    data = np.squeeze(hdul[0].data)
    wcs = WCS(header).celestial
    if data.ndim > 2: data = data[0] if data.ndim == 3 else data[0,0]
    
    try: pscale = abs(wcs.wcs.cdelt[1])
    except: pscale = 1.0/3600.0
    beam = get_beam_info(header, pscale)
    
    f_isl = os.path.join(out_dir, f"{args.prefix}_islands.csv")
    f_comp = os.path.join(out_dir, f"{args.prefix}_components.csv")
    
    if cfg['mosaic']:
        run_mosaic(data, wcs, beam, pscale, cfg, f_isl, f_comp, out_dir)
    else:
        run_standard(data, wcs, beam, pscale, cfg, f_isl, f_comp, out_dir)
    
    if DIAGNOSTICS_AVAILABLE:
        print("\n--- Running Diagnostics Module ---")
        try:
            diag = DiagnosticsRunner(f_comp, f_isl, out_dir)
            diag.run_all()
        except Exception as e: print(f"Error: {e}")
            
    print(f"\nPipeline Finished. Results in: {out_dir}")
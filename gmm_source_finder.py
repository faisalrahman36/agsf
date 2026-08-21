'''
Astronomy GMM Source Finder (AGSF)
Author: Syed Faisal ur Rahman
Version: 3 
'''

import argparse
import numpy as np
import warnings
import csv
import gc
import json
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.stats import SigmaClip
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_sources, SourceCatalog
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
import pandas as pd

warnings.filterwarnings('ignore')

DEFAULT_CONFIG = {
    "output_dir": "gmm_results",
    "save_plot": True,
    "mosaic": True,
    "tile_size": 2500,
    "padding": 100,
    "box_sizes": [50, 150, 400],
    "detection_sigma": 3.0,     # PyBDSF-style 3-sigma internal sub-component limit
    "peak_snr_sigma": 5.0,      # Strict 5-sigma global island limit
    "min_pix": 12,
    "n_jobs": -1,
    "exclusion_radius": 5.0,
    "max_components": 6,
    "multicomp_area_threshold": 3.0,
    "multicomp_snr_override": 15.0,
    "resample_factor": 100,
    "min_samples": 500,
    "max_samples": 10000,
    "resolved_sigma_threshold":2.5
}

_GAUSS_AREA_K = np.pi / (4.0 * np.log(2.0))

def setup_environment(config_path, cli_no_mosaic):
    cfg = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg.update(json.load(f))
    if cli_no_mosaic:
        cfg['mosaic'] = False
    out_dir = cfg.get('output_dir', 'gmm_output')
    os.makedirs(out_dir, exist_ok=True)
    return cfg, out_dir

def get_beam_info(header, pixel_scale):
    try:
        bmaj = header['BMAJ']
        bmin = header['BMIN']
        bpa = header.get('BPA', 0.0)
        beam_area = _GAUSS_AREA_K * (bmaj / pixel_scale) * (bmin / pixel_scale)
        return bmaj, bmin, bpa, beam_area
    except KeyError:
        return 1.0/3600, 1.0/3600, 0.0, 1.0

def deconvolve(maj, min_ax, pa, bmaj, bmin, bpa):
    """Astronomical PA matrix convention (North through East)."""
    pa_rad = np.radians(pa)
    bpa_rad = np.radians(bpa)
    fwhm_to_sigma = 1.0 / 2.355
    
    def get_cov(m_val, min_v, p_v):
        sx = (m_val * fwhm_to_sigma)**2
        sy = (min_v * fwhm_to_sigma)**2
        cp = np.cos(p_v)
        sp = np.sin(p_v)
        R = np.array([[sp, cp], [-cp, sp]])
        return R.T @ np.diag([sx, sy]) @ R
        
    Cov_int = get_cov(maj, min_ax, pa_rad) - get_cov(bmaj, bmin, bpa_rad)
    evals, evecs = np.linalg.eigh(Cov_int)
    if np.any(evals <= 0):
        return 0.0, 0.0, 0.0
        
    order = evals.argsort()[::-1]
    dc_maj = np.sqrt(evals[order[0]]) * 2.355
    dc_min = np.sqrt(evals[order[1]]) * 2.355
    dc_pa = np.degrees(np.arctan2(evecs[0, order[0]], evecs[1, order[0]])) % 180
    return dc_maj, dc_min, dc_pa

def calculate_errors(fp, fi, maj, min_ax, pa, bmaj, bmin, snr):
    """
    Condon (1997) analytical errors using Effective SNR.
    FIX: Now accepts 'pa' to project positional errors isotropically.
    """
    snr = max(snr, 0.1)
    rho_sq = max((maj * min_ax / (bmaj * bmin)) * (snr**2), 1.0) 

    em = maj * np.sqrt(2.0 / rho_sq)
    en = min_ax * np.sqrt(2.0 / rho_sq)

    denom = maj**2 - min_ax**2
    if denom < 1e-10:
        epa = 90.0 
    else:
        epa_rad = np.sqrt(4.0 / rho_sq) * (maj * min_ax / denom)
        epa = min(np.degrees(epa_rad), 90.0) 

    # Astrometric Fix: Project axes onto RA/DEC grid using PA
    pos_factor = np.sqrt(rho_sq * 8.0 * np.log(2.0))
    e_maj_axis = maj / pos_factor
    e_min_axis = min_ax / pos_factor

    pa_rad = np.radians(pa)
    era = np.sqrt((e_maj_axis * np.sin(pa_rad))**2 + (e_min_axis * np.cos(pa_rad))**2)
    edec = np.sqrt((e_maj_axis * np.cos(pa_rad))**2 + (e_min_axis * np.sin(pa_rad))**2)

    # 1% calibration floor added to empirical photometric errors
    ep = fp * np.sqrt(2.0 / rho_sq + 0.01**2)
    ei = fi * np.sqrt(2.0 / rho_sq + (em / maj)**2 + (en / min_ax)**2 + 0.01**2)

    return ep, ei, em, en, epa, era, edec

def fit_island_worker(task):
    island_id = task['id']
    cutout = task['cutout']
    mask = task['mask']
    wcs_slice = task['wcs']
    beam = task['beam']
    pix_scale = task['pix_scale']
    config = task['config']
    box_origin = task.get('box_origin', 0)
    rms = task['rms']

    if rms == 0 or np.isnan(rms):
        return []

    valid = (mask) & (~np.isnan(cutout))
    y, x = np.indices(cutout.shape)
    flux_vals = cutout[valid]
    
    if len(flux_vals) < 5 or np.sum(flux_vals) <= 0:
        return []

    X = np.vstack([x[valid], y[valid]]).T
    bmaj, bmin, bpa, beam_area = beam
    
    # Physical flux in this island for Top-Down partitioning
    island_flux_total = np.sum(flux_vals) / beam_area

    # Stochastic Sampling 
    prob = flux_vals / np.sum(flux_vals)
    prob = np.clip(prob, 0, None)
    prob /= np.sum(prob)
    
    n_samples = int(np.sum(flux_vals) * config.get('resample_factor', 100))
    n_samples = min(max(n_samples, config.get('min_samples', 500)), config.get('max_samples', 10000))
    
    try:
        X_resampled = X[np.random.choice(len(X), size=n_samples, p=prob)]
    except ValueError:
        return []

    island_snr = np.max(flux_vals) / rms
    lim_t = config.get('multicomp_area_threshold', 3.0) * beam_area
    max_comp = config['max_components'] if (len(flux_vals) >= lim_t or island_snr >= 15.0) else 1

    # Physical pixels effective sample size for BIC
    n_eff = max(len(X) / beam_area, 1.0)

    best_bic = np.inf
    best_gmm = None

    for n in range(1, max_comp + 1):
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full', 
                                  reg_covar=1e-6, random_state=42, n_init=3)
            gmm.fit(X_resampled)
            
            # Re-scale the log-likelihood to physical pixels to allow aggressive deblending
            cur_bic = gmm.bic(X_resampled)

            if cur_bic < best_bic:
                best_bic = cur_bic
                best_gmm = gmm
        except Exception as e:
            continue

    if not best_gmm:
        return []

    comps = []
    for i in range(best_gmm.n_components):
        mx, my = best_gmm.means_[i]
        vals, vecs = np.linalg.eigh(best_gmm.covariances_[i])
        order = vals.argsort()[::-1]
        
        maj_deg = np.sqrt(vals[order[0]]) * 2.355 * pix_scale
        min_deg = np.sqrt(vals[order[1]]) * 2.355 * pix_scale
        pa = np.degrees(np.arctan2(vecs[0, order[0]], vecs[1, order[0]])) % 180

        cw = best_gmm.weights_[i]
        dc_maj, dc_min, dc_pa = deconvolve(maj_deg, min_deg, pa, bmaj, bmin, bpa)

        # TOP-DOWN 
        int_f = island_flux_total * cw
        snr_est = int_f / rms if rms > 0 else 0.1
        
        resolved_sigma_threshold = config["resolved_sigma_threshold"]
        # Condon Resolution Envelope
        rho_sq_est = max((maj_deg * min_deg / (bmaj * bmin)) * (snr_est**2), 1.0)
        sigma_maj_est = maj_deg * np.sqrt(2.0 / rho_sq_est)
        is_resolved_maj = maj_deg >= (bmaj + resolved_sigma_threshold * sigma_maj_est)

        # Derive Peak Flux mathematically to prevent ghost components
        if not is_resolved_maj or min_deg < bmin or dc_maj == 0.0:
            maj_deg, min_deg, pa = bmaj, bmin, bpa
            dc_maj, dc_min, dc_pa = 0.0, 0.0, 0.0
            peak_f_comp = int_f  
        else:
            g_area_pixels = _GAUSS_AREA_K * (maj_deg / pix_scale) * (min_deg / pix_scale)
            peak_f_comp = int_f * (beam_area / g_area_pixels)

        #  sub-component threshold (3-sigma as default)
        if peak_f_comp < config.get('detection_sigma', 3.0) * rms:
            continue

        # Pass 'pa' into calculate_errors for RA/DEC projection
        errs = calculate_errors(peak_f_comp, int_f, maj_deg, min_deg, pa, bmaj, bmin, snr_est)

        try:
            sky = wcs_slice.pixel_to_world(mx, my)
            comp_data = {
                'Island_id': island_id, 'RA': sky.ra.deg, 'E_RA': errs[5], 'DEC': sky.dec.deg, 'E_DEC': errs[6],
                'Total_flux': int_f, 'E_Total_flux': errs[1], 'Peak_flux': peak_f_comp, 'E_Peak_flux': errs[0],
                'Maj': maj_deg*3600, 'E_Maj': errs[2]*3600, 'Min': min_deg*3600, 'E_Min': errs[3]*3600,
                'PA': pa, 'E_PA': errs[4], 'DC_Maj': dc_maj*3600, 'DC_Min': dc_min*3600, 'DC_PA': dc_pa,
                'RMS': rms, 'S_Code': 'S' if best_gmm.n_components == 1 else 'C', 'Detection_Box': box_origin
            }
            comps.append(comp_data)
        except Exception as e:
            continue

    return comps

def generate_production_plot(data, wcs, components, output_path, pixel_scale):
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(projection=wcs)
    norm = ImageNormalize(data, interval=ZScaleInterval())
    ax.imshow(data, origin='lower', cmap='Greys', norm=norm)
    for c in components:
        sky = SkyCoord(c['RA'], c['DEC'], unit='deg')
        x_pix, y_pix = wcs.world_to_pixel(sky)
        e = Ellipse((x_pix, y_pix), width=(c['Min']/3600)/pixel_scale, height=(c['Maj']/3600)/pixel_scale, angle=90+c['PA'], edgecolor='red', facecolor='none', lw=0.8)
        ax.add_patch(e)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_candidates(cands, beam, scale, config, f_isl, f_comp, group_id, append=False):
    if not cands:
        return []
    
    fit_tasks = [{'id': f"{group_id}_{i+1}", 'cutout': c['cutout'], 'mask': c['mask'], 'rms': c['rms'], 'wcs': c['wcs'], 'beam': beam, 'pix_scale': scale, 'config': config, 'box_origin': c['box']} for i, c in enumerate(cands)]
    results = Parallel(n_jobs=config['n_jobs'])(delayed(fit_island_worker)(t) for t in fit_tasks)
    
    comps = []
    for sublist in results:
        comps.extend(sublist)
        
    island_list = []
    for i, cand in enumerate(cands):
        uid = f"{group_id}_{i+1}"
        isl_comps = [c for c in comps if c['Island_id'] == uid]
        
        if not isl_comps: continue
            
        tot_f = sum([c['Total_flux'] for c in isl_comps])
        ra_avg = sum([c['RA'] * c['Total_flux'] for c in isl_comps]) / tot_f
        dec_avg = sum([c['DEC'] * c['Total_flux'] for c in isl_comps]) / tot_f
        e_ra = np.sqrt(sum([(c['E_RA'] * c['Total_flux'])**2 for c in isl_comps])) / tot_f
        e_dec = np.sqrt(sum([(c['E_DEC'] * c['Total_flux'])**2 for c in isl_comps])) / tot_f
        maj_v = cand['maj_sig'] if not np.isnan(cand['maj_sig']) else 0.0
        min_v = cand['min_sig'] if not np.isnan(cand['min_sig']) else 0.0
        pa_v = (90 - np.degrees(cand['orient'])) % 180 if not np.isnan(cand['orient']) else 0.0
        
        island_list.append({
            'Island_id': uid, 'RA': ra_avg, 'E_RA': e_ra, 'DEC': dec_avg, 'E_DEC': e_dec,
            'Total_flux': tot_f, 'E_Total_flux': np.sqrt(sum([c['E_Total_flux']**2 for c in isl_comps])),
            'Peak_flux': max([c['Peak_flux'] for c in isl_comps]), 'E_Peak_flux': max([c['E_Peak_flux'] for c in isl_comps]),
            'Maj': (maj_v*2.355*scale)*3600, 'Min': (min_v*2.355*scale)*3600, 'PA': pa_v,
            'Isl_rms': cand['rms'], 'Detection_Box': cand['box']
        })

    for f_path, data in [(f_isl, island_list), (f_comp, comps)]:
        if not data: continue
        mode = 'a' if append else 'w'
        write_header = not append or not os.path.exists(f_path) or os.path.getsize(f_path) == 0
        with open(f_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            if write_header: writer.writeheader()
            writer.writerows(data)
    return comps

def detect_on_data(data, wcs, config, edge_info=None):
    all_candidates = []
    h, w = data.shape
    nan_mask = np.isnan(data)
    
    for box in config['box_sizes']:
        try:
            bkg = Background2D(data, (box, box), filter_size=(3, 3), sigma_clip=SigmaClip(sigma=3.0), bkg_estimator=MedianBackground(), coverage_mask=nan_mask, fill_value=0.0)
            sub = data - bkg.background
            rms = bkg.background_rms
            
            segm = detect_sources(sub, config['detection_sigma'] * rms, npixels=config['min_pix'], mask=nan_mask)
            if segm is None: continue
                
            cat = SourceCatalog(sub, segm, error=rms, wcs=wcs, mask=nan_mask)
            for s in cat:
                l_rms = rms[int(s.centroid[1]), int(s.centroid[0])]
                if s.max_value < (config['peak_snr_sigma'] * l_rms): continue
                if edge_info:
                    il, ir, ib, it = edge_info
                    p = config['padding']
                    if (not il and s.centroid[0] < p) or (not ir and s.centroid[0] > w - p) or (not ib and s.centroid[1] < p) or (not it and s.centroid[1] > h - p): continue
                all_candidates.append({
                    'cx': s.centroid[0], 'cy': s.centroid[1], 'flux': s.segment_flux, 'peak': s.max_value,
                    'maj_sig': getattr(s.semimajor_sigma, 'value', s.semimajor_sigma), 'min_sig': getattr(s.semiminor_sigma, 'value', s.semiminor_sigma),
                    'orient': getattr(s.orientation, 'value', s.orientation), 'cutout': sub[s.slices].copy(),
                    'mask': (segm.data[s.slices] == s.label), 'rms': l_rms, 'wcs': wcs.slice(s.slices), 'box': box
                })
        except Exception as e:
            print(f"Background estimation failed for box {box}: {e}")
            continue
            
    if not all_candidates: return [], None
    all_candidates.sort(key=lambda x: x['flux'], reverse=True)
    unique_cands = []
    excl_sq = config['exclusion_radius']**2
    while all_candidates:
        curr = all_candidates.pop(0)
        unique_cands.append(curr)
        all_candidates = [c for c in all_candidates if ((c['cx']-curr['cx'])**2 + (c['cy']-curr['cy'])**2) > excl_sq]
    return unique_cands, None

def run_mosaic(data, full_wcs, beam, scale, config, f_isl, f_comp, out_dir):
    ts = config['tile_size']
    pad = config['padding']
    h, w = data.shape
    all_comps = []
    
    open(f_isl, 'a').close()
    open(f_comp, 'a').close()
    
    for y in range(0, h, ts):
        for x in range(0, w, ts):
            x0 = max(0, x - pad)
            x1 = min(w, x + ts + pad)
            y0 = max(0, y - pad)
            y1 = min(h, y + ts + pad)
            
            print(f"Processing Tile: T{x}_{y}...")
            edge_info = (x0 == 0, x1 == w, y0 == 0, y1 == h)
            cands, _ = detect_on_data(data[y0:y1, x0:x1], full_wcs[y0:y1, x0:x1], config, edge_info)
            tc = process_candidates(cands, beam, scale, config, f_isl, f_comp, f"T{x}_{y}", append=True)
            if tc: all_comps.extend(tc)
            gc.collect()
            
    print("Mosaic Processing Complete.")
    if config['save_plot'] and all_comps and data.size < 8000**2:
        generate_production_plot(data, full_wcs, all_comps, os.path.join(out_dir, "mosaic_overview.png"), scale)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("fits_file")
    p.add_argument("--prefix", default="gmm")
    p.add_argument("--config", default="config.json")
    p.add_argument("--no-mosaic", action="store_true")
    args = p.parse_args()

    cfg, out_dir = setup_environment(args.config, args.no_mosaic)

    hdul = fits.open(args.fits_file, memmap=True)
    header = hdul[0].header
    data = np.squeeze(hdul[0].data)
    wcs = WCS(header).celestial
    

    # --- METADATA CHECK BLOCK ---
    freq_hz = None
    # Check common FITS keywords for frequency (Rest Frequency or 3rd Axis)
    for key in ['RESTFRQ', 'RESTFREQ', 'CRVAL3']:
        if key in header:
            freq_hz = header[key]
            break
            
    print("\n" + "="*50)
    print("--- METADATA & PROVENANCE CHECK ---")
    print(f"Filename: {args.fits_file}")
    if freq_hz:
        freq_mhz = freq_hz / 1e6
        print(f"Header Frequency Detected: {freq_mhz:.3f} MHz")
        
    else:
        print("[!] WARNING: Could not detect Rest Frequency in FITS header.")
    print("="*50 + "\n")
    # -------------------------------------
    
    if data.ndim > 2:
        data = data[0] if data.ndim == 3 else data[0, 0]

    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        pscale = float(proj_plane_pixel_scales(wcs)[1])
    except Exception:
        try:
            pscale = abs(wcs.wcs.cdelt[1])
        except AttributeError:
            pscale = abs(wcs.pixel_scale_matrix[1, 1])

    beam = get_beam_info(header, pscale)
    
    f_isl = os.path.join(out_dir, f"{args.prefix}_islands.csv")
    f_comp = os.path.join(out_dir, f"{args.prefix}_components.csv")
    
    with open(f_isl, 'w', newline='') as f:
        csv.writer(f).writerow(['Island_id', 'RA', 'E_RA', 'DEC', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Maj', 'Min', 'PA', 'Isl_rms', 'Detection_Box'])
    with open(f_comp, 'w', newline='') as f:
        csv.writer(f).writerow(['Island_id', 'RA', 'E_RA', 'DEC', 'E_DEC', 'Total_flux', 'E_Total_flux', 'Peak_flux', 'E_Peak_flux', 'Maj', 'E_Maj', 'Min', 'E_Min', 'PA', 'E_PA', 'DC_Maj', 'DC_Min', 'DC_PA', 'RMS', 'S_Code', 'Detection_Box'])

    if cfg['mosaic']:
        run_mosaic(data, wcs, beam, pscale, cfg, f_isl, f_comp, out_dir)
    else:
        cands, _ = detect_on_data(data, wcs, cfg)
        tc = process_candidates(cands, beam, pscale, cfg, f_isl, f_comp, "FullFrame")
        if cfg['save_plot'] and tc:
            generate_production_plot(data, wcs, tc, os.path.join(out_dir, "full_frame_plot.png"), pscale)
            
    hdul.close()

    # deduplication
    try:
        
        
        def global_deduplicate(file_path, exclusion_radius_arcsec):
            print(f"\n[+] Running global cross-tile deduplication on {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)
            initial_count = len(df)
            
            if initial_count < 2:
                return
                
            # Match catalog to itself using astropy's 2nd nearest neighbor (1st is itself)
            coords = SkyCoord(ra=df['RA'].values, dec=df['DEC'].values, unit='deg')
            idx, d2d, _ = match_coordinates_sky(coords, coords, nthneighbor=2)
            
            keep = np.ones(initial_count, dtype=bool)
            
            # Iterate through any matches closer than the exclusion radius
            for j in np.where(d2d.arcsec < exclusion_radius_arcsec)[0]:
                partner = idx[j]
                # If both are still flagged to be kept, a duplicate overlap exists
                if keep[j] and keep[partner]:
                    # Keep the fragment with the higher Peak_flux, drop the duplicate
                    if df['Peak_flux'].iloc[j] < df['Peak_flux'].iloc[partner]:
                        keep[j] = False
                    else:
                        keep[partner] = False
                        
            df_cleaned = df[keep]
            final_count = len(df_cleaned)
            df_cleaned.to_csv(file_path, index=False)
            print(f"[SUCCESS] Removed {initial_count - final_count} overlapping mosaic duplicates. Final count: {final_count}")

        # Execute deduplication using the config exclusion radius converted to arcseconds
        dedup_radius_arcsec = cfg['exclusion_radius'] * pscale * 3600
        
        global_deduplicate(f_comp, dedup_radius_arcsec)
        global_deduplicate(f_isl, dedup_radius_arcsec)
        
    except ImportError:
        print("[!] Error: 'pandas' is not installed. Global deduplication skipped. Run 'pip install pandas'.")
    except Exception as e:
        print(f"[!] CRITICAL ERROR during deduplication: {e}")
        print("[!] Proceeding without deduplication. Output catalog may contain flux-inflated mosaic boundary duplicates.")
    
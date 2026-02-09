#!/usr/bin/env python3
# select_and_run_with_autobalance_v2.py
"""
Refactored NLFFF Automation Script (NumPy 2.0 Compatible)
Improvements:
- Fixed np.trapz error (uses np.trapezoid)
- Non-periodic finite difference derivatives (np.gradient)
- Weighted Flux Balancing
- Diffusive Divergence Cleaning in relaxation loop
- Trapezoidal integration for Energy with boundary apodization
"""

import os, sys, math, gc, json
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import warnings
from scipy import ndimage as ndi
from scipy.ndimage import zoom, gaussian_filter
import matplotlib.pyplot as plt

# Silence the WCS warning
warnings.simplefilter('ignore', category=AstropyWarning)

# -------------------- SETTINGS --------------------
BR_FILE = "hmi.sharp_cea_720s.10346.20231103_013600_TAI.Br.fits"
BT_FILE = "hmi.sharp_cea_720s.10346.20231103_013600_TAI.Bt.fits"
BP_FILE = "hmi.sharp_cea_720s.10346.20231103_013600_TAI.Bp.fits"
BITMAP_FILE = "hmi.sharp_cea_720s.10346.20231103_013600_TAI.bitmap.fits"
CONT_FILE = "hmi.sharp_cea_720s.10346.20231103_013600_TAI.continuum.fits"

CSV_FILE = "SP3D20231104_210115.0C_resuuuuuults_results.csv"
OUT_DIR = "11out_nlfff_autobalance_v222"
os.makedirs(OUT_DIR, exist_ok=True)

# NLFFF / performance params
DOWNSAMPLE = 2       
NZ = 32
HEIGHT_Mm = 40.0
N_ITER = 300       
DT = 0.01          
NU = 0.2
ETA = 1e5          

# Divergence Cleaning Parameter
KAPPA_DIV = 0.1    

COMPONENT_THR = 50.0
PAD = 8
CSV_METHOD_DEFAULT = "max"
CSV_RADIUS_DEFAULT = 20.0
# -----------------------------------------------------------------

def load_fits_data(path):
    h = fits.open(path)
    if h[0].data is not None:
        data = np.array(h[0].data)
    else:
        data = np.array(h[1].data)
    hdr = h[0].header
    h.close()
    return data, hdr

print("Loading SHARP files...")
try:
    br_full, hdr = load_fits_data(BR_FILE)
    bt_full, _ = load_fits_data(BT_FILE)
    bp_full, _ = load_fits_data(BP_FILE)
    bitmap_full, _ = load_fits_data(BITMAP_FILE)
    cont_full, _ = load_fits_data(CONT_FILE)
except Exception as e:
    print("Error reading FITS:", e); sys.exit(1)

# -------------------- PRE-PROCESSING --------------------

# Connected components logic
harp_mask = (bitmap_full != 0).astype(int)
th_mask = ((np.abs(br_full) > COMPONENT_THR) & (harp_mask == 1)).astype(int)
labels, ncomp = ndi.label(th_mask)
if ncomp == 0:
    labels, ncomp = ndi.label(harp_mask)
print("Found connected components:", ncomp)

components = []
slices = ndi.find_objects(labels)
for lab in range(1, np.max(labels)+1):
    loc = np.where(labels == lab)
    if loc[0].size == 0: continue
    area = loc[0].size
    cy = float(np.mean(loc[0])); cx = float(np.mean(loc[1]))
    mean_absB = float(np.mean(np.abs(br_full[labels == lab])))
    mean_cont = float(np.mean(cont_full[labels == lab]))
    components.append({'label': lab, 'area': int(area), 'cx': cx, 'cy': cy,
                       'mean_absB': mean_absB, 'mean_cont': mean_cont})
components_sorted_by_y = sorted(components, key=lambda c: c['cy'])

# Save components CSV
csv_out = os.path.join(OUT_DIR, "components_list.csv")
import csv
with open(csv_out, "w", newline='') as cf:
    w = csv.writer(cf)
    w.writerow(['label','area','cx','cy','mean_absB','mean_cont'])
    for c in components_sorted_by_y:
        w.writerow([c['label'], c['area'], c['cx'], c['cy'], c['mean_absB'], c['mean_cont']])

# Overview image
plt.figure(figsize=(11,6))
plt.imshow(cont_full, origin='lower', cmap='gray')
components_by_area = sorted(components, key=lambda c: c['area'], reverse=True)
labels_to_show = set([c['label'] for c in components_by_area[:60]])
for c in components_sorted_by_y:
    if c['label'] in labels_to_show:
        plt.text(c['cx'], c['cy'], str(c['label']), color='yellow', fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.2'))
plt.title("Continuum overview")
plt.savefig(os.path.join(OUT_DIR, "components_overview_labeled.png"), dpi=200); plt.close()

# -------------------- USER SELECTION LOGIC --------------------

def detect_csv_columns(df):
    cols_lower = [c.lower() for c in df.columns]
    xcol = ycol = bcol = None
    for cand in ('x_arcsec','x_arc','x','x_pix','hpc_x'):
        if cand in cols_lower: xcol = df.columns[cols_lower.index(cand)]; break
    for cand in ('y_arcsec','y_arc','y','y_pix','hpc_y'):
        if cand in cols_lower: ycol = df.columns[cols_lower.index(cand)]; break
    for cand in ('b_g','b_g_gauss','bgauss','b','bg','b_field','b_wfa','b_wfa_g','b_mc_mean'):
        if cand in cols_lower: bcol = df.columns[cols_lower.index(cand)]; break
    if bcol is None:
        for i,c in enumerate(cols_lower):
            if 'b' in c and ('g' in c or 'gauss' in c or 'wfa' in c or 'mc' in c):
                bcol = df.columns[i]; break
    return xcol, ycol, bcol

csv_has=False; csv_df=None; xcol=ycol=bcol=None
if CSV_FILE and os.path.exists(CSV_FILE):
    try:
        csv_df = pd.read_csv(CSV_FILE)
        csv_has = True
        xcol,ycol,bcol = detect_csv_columns(csv_df)
        print("Loaded CSV columns:", xcol, ycol, bcol)
    except: pass

def pick_B_from_csv(csv_df, xcol, ycol, bcol, x0, y0, method="max", radius_arcsec=20.0):
    info = {"method":method, "radius":radius_arcsec, "coord":(x0,y0)}
    x = np.asarray(csv_df[xcol].astype(float))
    y = np.asarray(csv_df[ycol].astype(float))
    b = np.asarray(csv_df[bcol].astype(float))
    d2 = (x - x0)**2 + (y - y0)**2
    idx = np.where(d2 <= radius_arcsec**2)[0]
    if idx.size == 0:
        idxn = int(np.argmin(d2))
        return float(b[idxn]), info
    bs = b[idx]; ds = np.sqrt(d2[idx])
    if method=="nearest": val = float(bs[np.argmin(ds)])
    elif method=="max": val = float(bs[np.argmax(np.abs(bs))])
    elif method=="median": val = float(np.nanmedian(bs))
    elif method=="mean": val = float(np.nanmean(bs))
    elif method=="weighted":
        w = 1.0/(ds+1e-3); val = float(np.sum(bs*w)/np.sum(w))
    else: val = float(bs[np.argmax(np.abs(bs))])
    return val, info

print("\nChoose how to select region:")
print(" 1) Use CSV coordinate")
print(" 2) Enter arcsec coordinate")
print(" 3) Click on image")
print(" 4) Auto-top")
choice = input("Select [default 4]: ").strip() or "4"

selected_B_value = None; selected_coord_arcsec=None; selected_csv_info=None
ny_full, nx_full = br_full.shape

if choice=="1" and csv_has:
    ref = input("Enter 'x y' (arcsec) or Enter for auto: ").strip()
    if ref: 
        try: selected_coord_arcsec = tuple(map(float, ref.split()))
        except: pass
    
    if selected_coord_arcsec:
        val, info = pick_B_from_csv(csv_df, xcol, ycol, bcol, *selected_coord_arcsec, method=CSV_METHOD_DEFAULT, radius_arcsec=CSV_RADIUS_DEFAULT)
        selected_B_value = val; selected_csv_info = info
    else:
        # Global max fallback
        bvals = csv_df[bcol].astype(float)
        selected_B_value = float(bvals.iloc[np.argmax(np.abs(bvals))])

elif choice=="2":
    s = input("Enter 'x y': ").strip()
    selected_coord_arcsec = tuple(map(float, s.split()))
elif choice=="3":
    print("Click on window...")
    fig,ax=plt.subplots(); ax.imshow(cont_full,origin='lower',cmap='gray')
    pts=plt.ginput(1); plt.close(fig)
    if pts:
        w=WCS(hdr); world=w.wcs_pix2world([pts[0]],0)[0]
        selected_coord_arcsec=(float(world[0])*3600, float(world[1])*3600)
elif choice=="4":
    chosen_label = components_sorted_by_y[0]['label']

# Resolve label
if selected_coord_arcsec and choice != "4":
    try:
        w = WCS(hdr)
        pix = w.wcs_world2pix([[selected_coord_arcsec[0]/3600, selected_coord_arcsec[1]/3600]],0)[0]
        r, c = int(round(pix[1])), int(round(pix[0]))
        if 0<=r<ny_full and 0<=c<nx_full:
            lab = labels[r,c]
            if lab!=0: chosen_label=lab
            else: chosen_label = components_sorted_by_y[0]['label'] # Fallback
        else: chosen_label = components_sorted_by_y[0]['label']
    except:
        chosen_label = components_sorted_by_y[0]['label']
elif choice=="4":
    chosen_label = components_sorted_by_y[0]['label']

if 'chosen_label' not in locals(): chosen_label = components_sorted_by_y[0]['label']

# -------------------- CROPPING & PREP --------------------

sl = slices[chosen_label-1]
y0 = max(0, sl[0].start - PAD); y1 = min(br_full.shape[0]-1, sl[0].stop + PAD - 1)
x0 = max(0, sl[1].start - PAD); x1 = min(br_full.shape[1]-1, sl[1].stop + PAD - 1)
br_crop = br_full[y0:y1+1, x0:x1+1].astype(np.float32)
bt_crop = bt_full[y0:y1+1, x0:x1+1].astype(np.float32)
bp_crop = bp_full[y0:y1+1, x0:x1+1].astype(np.float32)
mask_crop = (labels[y0:y1+1,x0:x1+1]==chosen_label).astype(float)

if DOWNSAMPLE > 1:
    br_crop = zoom(br_crop, 1/DOWNSAMPLE, order=1)
    bt_crop = zoom(bt_crop, 1/DOWNSAMPLE, order=1)
    bp_crop = zoom(bp_crop, 1/DOWNSAMPLE, order=1)
    mask_crop = zoom(mask_crop, 1/DOWNSAMPLE, order=0)

mask_smooth = gaussian_filter(mask_crop, sigma=2)
if mask_smooth.max()>0: mask_smooth /= mask_smooth.max()

# -------------------- PHYSICS HELPERS (Refactored) --------------------

def derivative_3d(f, dx, dy, dz):
    """
    Computes derivatives using 2nd order central differences (np.gradient).
    Handles non-periodic boundaries correctly.
    """
    gz = np.gradient(f, dz, axis=0, edge_order=2)
    gy = np.gradient(f, dy, axis=1, edge_order=2)
    gx = np.gradient(f, dx, axis=2, edge_order=2)
    return gx, gy, gz

def compute_curl(Bx, By, Bz, dx, dy, dz):
    """
    Calculates curl(B) using non-periodic finite differences.
    J = curl(B)
    """
    Bxx, Bxy, Bxz = derivative_3d(Bx, dx, dy, dz)
    Byx, Byy, Byz = derivative_3d(By, dx, dy, dz)
    Bzx, Bzy, Bzz = derivative_3d(Bz, dx, dy, dz)
    
    Jx = Bzy - Byz
    Jy = Bxz - Bzx
    Jz = Byx - Bxy
    return Jx, Jy, Jz

def compute_divergence(Bx, By, Bz, dx, dy, dz):
    """ Calculates div(B) """
    Bxx, _, _ = derivative_3d(Bx, dx, dy, dz)
    _, Byy, _ = derivative_3d(By, dx, dy, dz)
    _, _, Bzz = derivative_3d(Bz, dx, dy, dz)
    return Bxx + Byy + Bzz

def potential_field_alissandrakis_padded(bz0, nz, height_Mm):
    """
    Alissandrakis FFT solver. 
    NOTE: Input bz0 is padded to mitigate periodic boundary effects for the potential calculation.
    """
    # Pad to power of 2 for efficiency and isolation
    ny, nx = bz0.shape
    pad_y = ny // 4
    pad_x = nx // 4
    bz_pad = np.pad(bz0, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    
    ny_p, nx_p = bz_pad.shape
    
    # FFT
    bz_k = np.fft.fft2(bz_pad)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx_p, d=1.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny_p, d=1.0)
    kxg, kyg = np.meshgrid(kx, ky)
    kperp = np.sqrt(kxg**2 + kyg**2)
    
    z_cm = np.linspace(0.0, height_Mm * 1e8, nz)
    
    # Result arrays (padded size)
    Bx_p = np.zeros((nz, ny_p, nx_p), dtype=np.float32)
    By_p = np.zeros_like(Bx_p)
    Bz_p = np.zeros_like(Bx_p)
    
    tiny = 1e-20
    for iz, z in enumerate(z_cm):
        expkz = np.exp(-kperp * z)
        # Bz component
        Bz_kz = bz_k * expkz
        # Potential phi: B = -grad phi -> B_k = -i k phi_k
        # div B = 0 -> d^2phi/dz^2 - k^2 phi = 0
        # Solution leads to standard alpha=0 relations
        # For potential: Bz_k = kperp * phi_k  => phi_k = Bz_k / kperp
        # Bx_k = -i kx phi_k = -i kx (Bz_k / kperp)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            factor = -1j / (kperp + tiny)
            Bx_kz = kxg * factor * Bz_kz
            By_kz = kyg * factor * Bz_kz
        
        # Handle k=0 (mean field) - usually 0 for balanced flux
        Bx_kz[0,0] = 0; By_kz[0,0] = 0
        
        Bx_p[iz] = np.real(np.fft.ifft2(Bx_kz))
        By_p[iz] = np.real(np.fft.ifft2(By_kz))
        Bz_p[iz] = np.real(np.fft.ifft2(Bz_kz))
        
    # Crop back to original size
    return (Bx_p[:, pad_y:-pad_y, pad_x:-pad_x], 
            By_p[:, pad_y:-pad_y, pad_x:-pad_x], 
            Bz_p[:, pad_y:-pad_y, pad_x:-pad_x])

def magneto_frictional_solver(Bx, By, Bz, dx, dy, dz, nu, eta, kappa_div, dt, niter, 
                              Bx_bot, By_bot, Bz_bot):
    """
    Relaxation with Divergence Cleaning and Non-Periodic Derivatives.
    """
    for it in range(niter):
        # 1. Calculate Lorentz Force components
        Jx, Jy, Jz = compute_curl(Bx, By, Bz, dx, dy, dz)
        
        # Lorentz force F = J x B
        Fx = Jy*Bz - Jz*By
        Fy = Jz*Bx - Jx*Bz
        Fz = Jx*By - Jy*Bx
        
        B2 = Bx**2 + By**2 + Bz**2 + 1e-8
        
        # Velocity v = F / (nu * B^2)
        v_factor = 1.0 / (nu * B2)
        vx = Fx * v_factor
        vy = Fy * v_factor
        vz = Fz * v_factor
        
        # 2. Induction Equation: dB/dt = curl(v x B) + eta*Lap(B) + kappa*grad(div B)
        # v x B
        vxB_x = vy*Bz - vz*By
        vxB_y = vz*Bx - vx*Bz
        vxB_z = vx*By - vy*Bx
        
        curl_vxB_x, curl_vxB_y, curl_vxB_z = compute_curl(vxB_x, vxB_y, vxB_z, dx, dy, dz)
        
        # Laplacian (Diffusion) via gradients
        lap_Bx = compute_divergence(*derivative_3d(Bx, dx, dy, dz), dx, dy, dz)
        lap_By = compute_divergence(*derivative_3d(By, dx, dy, dz), dx, dy, dz)
        lap_Bz = compute_divergence(*derivative_3d(Bz, dx, dy, dz), dx, dy, dz)

        # 3. Divergence Cleaning Term: grad(div B)
        divB = compute_divergence(Bx, By, Bz, dx, dy, dz)
        grad_divB_x, grad_divB_y, grad_divB_z = derivative_3d(divB, dx, dy, dz)
        
        # Update
        Bx += dt * (curl_vxB_x + eta * lap_Bx + kappa_div * grad_divB_x)
        By += dt * (curl_vxB_y + eta * lap_By + kappa_div * grad_divB_y)
        Bz += dt * (curl_vxB_z + eta * lap_Bz + kappa_div * grad_divB_z)
        
        # Boundary Conditions (Line-tying at bottom, open/zero-grad elsewhere)
        Bx[0,:,:] = Bx_bot
        By[0,:,:] = By_bot
        Bz[0,:,:] = Bz_bot
        
        # Simple open top/sides BCs (zero-gradient)
        Bx[-1,:,:] = Bx[-2,:,:]; By[-1,:,:] = By[-2,:,:]; Bz[-1,:,:] = Bz[-2,:,:]
        
        if (it+1) % 50 == 0:
            L_force = np.mean(np.sqrt(Fx**2 + Fy**2 + Fz**2))
            mean_div = np.mean(np.abs(divB))
            print(f" Iter {it+1}/{niter}: Mean Lorentz Force={L_force:.4e}, Mean DivB={mean_div:.4e}")
            
    return Bx, By, Bz

def compute_energy_robust(Bx, By, Bz, dx, dy, dz, pad=6):
    """
    Computes magnetic energy using Trapezoidal integration.
    Applies APODIZATION (padding) to ignore noisy boundaries.
    Detects NumPy version to use correct trapezoid function.
    """
    # Energy density
    E_dens = (Bx**2 + By**2 + Bz**2) / (8.0 * np.pi)
    
    # Apodize (exclude boundaries)
    if pad > 0 and Bx.shape[1] > 2*pad:
        E_dens = E_dens[:, pad:-pad, pad:-pad]
        
    # FIX: NumPy 2.0 removed np.trapz in favor of np.trapezoid
    if hasattr(np, 'trapezoid'):
        trapz_func = np.trapezoid
    else:
        trapz_func = np.trapz

    # Integrate Volume: dz, then dy, then dx
    E_z = trapz_func(E_dens, dx=dz, axis=0)
    E_zy = trapz_func(E_z, dx=dy, axis=0) # reduced dimension
    E_total = trapz_func(E_zy, dx=dx, axis=0)
    
    return float(E_total)

# -------------------- GENERATING VARIANTS --------------------

def make_bz_variant(bz_orig, mask, B_choice, variant):
    sub = bz_orig.copy().astype(np.float64)
    # Blend mask
    sign = np.sign(sub); sign[sign==0]=1.0
    replaced = sign * (np.abs(sub) * (1.0 - mask_smooth) + np.abs(B_choice) * mask_smooth)
    
    # 1. Standard handling
    if variant == 'orig': pass
    elif variant == 'zero_mean':
        replaced -= np.nanmean(replaced)
        
    elif variant.startswith('blend_'):
        alpha = float(variant.split('_')[1])
        zm = replaced - np.nanmean(replaced)
        replaced = (1.0-alpha)*replaced + alpha*zm
        
    # 2. IMPROVED FLUX BALANCING
    elif variant == 'flux_balance':
        # B_corr = B - (|B| / sum(|B|)) * net_flux
        # This distributes the flux error proportional to field strength, 
        # preserving weak features and zero-crossings.
        net_flux = np.nansum(replaced)
        abs_flux = np.nansum(np.abs(replaced))
        
        if abs_flux > 1e-9:
            correction = (np.abs(replaced) / abs_flux) * net_flux
            replaced -= correction
            
    # Dilate logic (simple morphology)
    elif variant.startswith('dilate_'):
        k = 2
        from scipy.ndimage import grey_dilation
        bigmask = grey_dilation(mask.astype(int), size=(k,k))
        ms = gaussian_filter(bigmask.astype(float), sigma=2); ms /= (ms.max() or 1)
        replaced = sign * (np.abs(sub) * (1.0 - ms) + np.abs(B_choice) * ms)
        # Apply balance to this too
        net = np.nansum(replaced); abs_f = np.nansum(np.abs(replaced))
        if abs_f > 1e-9: replaced -= (np.abs(replaced)/abs_f)*net

    return np.nan_to_num(replaced)

# Prepare candidates
candidates = {}
if selected_B_value and not math.isnan(selected_B_value):
    candidates['from_csv'] = float(selected_B_value)
try: candidates['mean_abs'] = float(np.nanmean(np.abs(br_crop)))
except: candidates['mean_abs'] = 0.0

small_files = {}
print("Generating B_z variants...")
for base, Bval in candidates.items():
    for v in ('orig','zero_mean','flux_balance','blend_0.5'):
        vname = f"{base}__{v}"
        bz_arr = make_bz_variant(br_crop, mask_crop, Bval, v)
        
        # Save small FITS
        out_s = os.path.join(OUT_DIR, f"small_{chosen_label}_{vname}.fits")
        fits.HDUList([
            fits.PrimaryHDU(bz_arr.astype(np.float32), header=hdr),
            fits.ImageHDU(bt_crop), fits.ImageHDU(bp_crop)
        ]).writeto(out_s, overwrite=True)
        small_files[vname] = bz_arr

# -------------------- RUNNING SIMULATIONS --------------------

# Physical dimensions
c1 = hdr.get('CDELT1') or 0.03
dx_cm = 6.957e10 * abs(float(c1)) * (np.pi/180) * DOWNSAMPLE
dz_cm = (HEIGHT_Mm * 1e8) / NZ

results = []

for vname, bz_in in small_files.items():
    print(f"\n>>> Processing {vname} ...")
    
    # Check flux balance
    f_net = np.sum(bz_in); f_abs = np.sum(np.abs(bz_in))
    ratio = f_net / f_abs if f_abs>0 else 1.0
    print(f"    Flux Ratio (Net/Abs): {ratio:.2e}")
    
    # 1. Potential Field (Initial Condition + Energy Ref)
    # Using padded solver to minimize boundary artifacts in Potential solution
    Bx_p, By_p, Bz_p = potential_field_alissandrakis_padded(bz_in, NZ, HEIGHT_Mm)
    
    # Compute Potential Energy using robust integration
    # Pad=PAD ensures we don't count boundary errors in energy
    E_pot = compute_energy_robust(Bx_p, By_p, Bz_p, dx_cm, dx_cm, dz_cm, pad=PAD)
    
    # 2. NLFFF Relaxation
    # Set bottom boundary
    Bx = Bx_p.copy(); By = By_p.copy(); Bz = Bz_p.copy()
    
    # Run Solver
    Bx_n, By_n, Bz_n = magneto_frictional_solver(
        Bx, By, Bz, 
        dx_cm, dx_cm, dz_cm, 
        NU, ETA, KAPPA_DIV, DT, N_ITER,
        bt_crop, bp_crop, bz_in
    )
    
    # 3. Compute NLFFF Energy
    E_nlfff = compute_energy_robust(Bx_n, By_n, Bz_n, dx_cm, dx_cm, dz_cm, pad=PAD)
    E_free = E_nlfff - E_pot
    
    print(f"    E_pot:   {E_pot:.6e}")
    print(f"    E_nlfff: {E_nlfff:.6e}")
    print(f"    E_free:  {E_free:.6e}  ({(E_free/E_pot)*100:.2f}%)")
    
    # Save 3D result
    out_3d = os.path.join(OUT_DIR, f"nlfff_{chosen_label}_{vname}.fits")
    fits.HDUList([
        fits.PrimaryHDU(Bx_n.astype(np.float32), header=hdr),
        fits.ImageHDU(By_n.astype(np.float32), name='By'),
        fits.ImageHDU(Bz_n.astype(np.float32), name='Bz')
    ]).writeto(out_3d, overwrite=True)
    
    results.append({
        "variant": vname,
        "E_pot": E_pot,
        "E_nlfff": E_nlfff,
        "E_free": E_free,
        "flux_ratio": ratio
    })

# -------------------- SELECTION & SUMMARY --------------------

# Select best: prefer Positive E_free > 0 and Low Flux Ratio
best = None
for r in results:
    if r['E_free'] > 0:
        if best is None or abs(r['flux_ratio']) < abs(best['flux_ratio']):
            best = r

if best is None:
    # Fallback to least negative E_free
    best = sorted(results, key=lambda x: abs(x['E_free']))[0]

print(f"\nBest Variant Selected: {best['variant']}")
print(f"Final E_free: {best['E_free']:.4e}")

with open(os.path.join(OUT_DIR, "summary_refactored.txt"), "w") as f:
    f.write(f"Refactored Run Summary\nBest: {best['variant']}\n\n")
    for r in results:
        f.write(str(r) + "\n")

print("Done.")

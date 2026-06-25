"""
=============================================================================
Retinal Prosthesis Simulator v4 — High-Resolution Output
=============================================================================

PIPELINE:
  image → YOLO crop → contour keypoints → current steering (6×10 grid)
  → deconvolution pre-emphasis → ScoreboardModel + AxonMapModel (xystep=0.05)
  → 6-panel figure saved to ~/Desktop/retinal_percept.png

NEUROSCIENCE:
  ScoreboardModel  — idealised round Gaussian phosphenes (rho=200µm)
  AxonMapModel     — biologically realistic axon streaks (rho=150µm, λ=100µm)
  Current Steering — bilinear amplitude split between 4 surrounding electrodes
                     → virtual phosphene at sub-grid position (Firszt 2007)
  Deconvolution    — unsharp-mask pre-emphasis to counteract retinal blurring
                     amp_out = amp + α(amp − Gaussian_σ(amp))

KEY PARAMETERS:
  xystep=0.05 dva/px  → 2× higher output resolution vs xystep=0.1
  ScoreboardModel: 317×525 px output (was 159×263)
  AxonMapModel:    uses xystep=0.06 for balance of detail vs speed
=============================================================================
"""

import os, sys, subprocess
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO
from pulse2percept.implants import ProsthesisSystem, ElectrodeArray, DiskElectrode
from pulse2percept.models import ScoreboardModel, AxonMapModel

# ════════════════════════════════════════════════════════════
#  PARAMETERS
# ════════════════════════════════════════════════════════════
ROWS, COLS           = 6, 10
ELECTRODE_SPACING_UM = 525        # µm  (Argus II equivalent)
ELECTRODE_RADIUS_UM  = 100        # µm

# ScoreboardModel
SCOREBOARD_RHO       = 200        # µm  phosphene radius (Gaussian sigma)
SCOREBOARD_XYSTEP    = 0.05       # dva/px — HIGH RES (was 0.1)

# AxonMapModel
AXON_RHO             = 150        # µm  phosphene width
AXON_LAMBDA          = 100        # µm  axon streak length (small = rounder)
AXON_XYSTEP          = 0.06       # dva/px — high res but slightly coarser for speed

MAX_AMP_UA           = 80         # µA  max stimulation amplitude
N_KEYPOINTS          = 30         # contour keypoints (boundary samples)
DECONV_ALPHA         = 0.5        # pre-emphasis strength  [0=off, 1=strong]
DECONV_SIGMA         = 0.8        # Gaussian sigma for pre-emphasis

CONF_THRESHOLD       = 0.3
OUTPUT_PATH          = os.path.expanduser("~/Desktop/retinal_percept.png")

# Visual field extent — derived from electrode spacing
# 1 dva ≈ 280 µm on human retina (Drasdo & Fowler 1974)
_hx = (COLS * ELECTRODE_SPACING_UM) / 2.0 / 280.0 * 1.4
_hy = (ROWS * ELECTRODE_SPACING_UM) / 2.0 / 280.0 * 1.4
MODEL_XRANGE = (-round(_hx, 1),  round(_hx, 1))   # ≈ ±13 dva
MODEL_YRANGE = (-round(_hy, 1),  round(_hy, 1))   # ≈ ±8 dva
# ════════════════════════════════════════════════════════════


# ── Build 6×10 electrode implant ────────────────────────────────────────
def build_implant():
    electrodes = {}
    for r in range(ROWS):
        for c in range(COLS):
            x = (c - (COLS-1)/2.0) * ELECTRODE_SPACING_UM
            y = (r - (ROWS-1)/2.0) * ELECTRODE_SPACING_UM
            electrodes[f"E{r}{c}"] = DiskElectrode(x, y, 0, ELECTRODE_RADIUS_UM)
    return ProsthesisSystem(earray=ElectrodeArray(electrodes))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — YOLO detect + crop
# ══════════════════════════════════════════════════════════════════════════
def detect_and_crop(image_path, yolo):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    results = yolo(frame, verbose=False)[0]
    best_conf, best_box, best_label = 0.0, None, "object"
    for det in results.boxes:
        c = float(det.conf)
        if c > best_conf and c >= CONF_THRESHOLD:
            best_conf  = c
            best_box   = det.xyxy[0].cpu().numpy()
            best_label = yolo.names[int(det.cls)]
    if best_box is None:
        h, w = frame.shape[:2]
        best_box = np.array([0, 0, w, h], dtype=float)
        best_label, best_conf = "object", 1.0
        print("  [WARN] YOLO found nothing — using full image")
    x1, y1, x2, y2 = [int(v) for v in best_box]
    return frame, frame[y1:y2, x1:x2], best_label, best_conf, best_box


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — CONTOUR KEYPOINTS
#
#  Object outer boundary → N evenly-spaced points → full 2D spread across grid
#  (Skeleton/medial-axis collapsed to a horizontal line — this was the v2 bug)
# ══════════════════════════════════════════════════════════════════════════
def extract_contour_keypoints(bgr_crop, n=N_KEYPOINTS):
    gray  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    blur  = cv2.GaussianBlur(gray, (7, 7), 0)

    # GrabCut segmentation
    mask   = np.zeros(gray.shape, np.uint8)
    margin = max(5, min(h, w) // 12)
    rect   = (margin, margin, w - 2*margin, h - 2*margin)
    bgd    = np.zeros((1, 65), np.float64)
    fgd    = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr_crop, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        fg = np.zeros_like(gray)

    # Fallback: Otsu threshold
    if fg.sum() < 1000:
        _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological clean-up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k)

    # Keep largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        fg = (labels == largest).astype(np.uint8) * 255

    # Find outer contour
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        edges = cv2.Canny(blur, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        cnt  = max(contours, key=cv2.contourArea)
        pts  = cnt.reshape(-1, 2).astype(float)   # (N, 2) in (x, y)
        pts  = pts[:, [1, 0]]                      # → (row, col)
    else:
        # Last resort: crop border
        border = ([(0, i) for i in range(w)] + [(h-1, i) for i in range(w)] +
                  [(i, 0) for i in range(h)] + [(i, w-1) for i in range(h)])
        pts = np.array(border, dtype=float)

    # Evenly subsample N keypoints
    idx  = np.round(np.linspace(0, len(pts)-1, n)).astype(int)
    kpts = pts[idx].copy()
    kpts[:, 0] /= (h - 1)
    kpts[:, 1] /= (w - 1)
    kpts = np.clip(kpts, 0.0, 1.0)

    # Visualisation
    contour_vis = bgr_crop.copy()
    if contours:
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 255), 2)

    return kpts, contour_vis, fg


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — CURRENT STEERING → 6×10 AMPLITUDE GRID
#
#  Bilinear interpolation: keypoint between 4 electrodes splits amplitude
#  proportionally → virtual phosphene at sub-grid position.
# ══════════════════════════════════════════════════════════════════════════
def keypoints_to_amplitudes(kpts):
    amp = np.zeros((ROWS, COLS), dtype=float)
    for yr, xc in kpts:
        rf = yr * (ROWS - 1)
        cf = xc * (COLS - 1)
        r0 = int(np.floor(rf));  r1 = min(r0+1, ROWS-1)
        c0 = int(np.floor(cf));  c1 = min(c0+1, COLS-1)
        dr = rf - r0;            dc = cf - c0
        amp[r0, c0] += (1-dr)*(1-dc) * MAX_AMP_UA
        amp[r0, c1] += (1-dr)* dc    * MAX_AMP_UA
        amp[r1, c0] +=    dr *(1-dc) * MAX_AMP_UA
        amp[r1, c1] +=    dr * dc    * MAX_AMP_UA
    if amp.max() > 0:
        amp = amp / amp.max() * MAX_AMP_UA
    return amp


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — PHOSPHENE DECONVOLUTION (pre-emphasis)
#
#  Retina blurs each electrode signal by Gaussian(ρ).
#  Unsharp-mask pre-sharpens amplitude grid to compensate:
#    amp_out = amp + α × (amp − Gaussian_σ(amp))
# ══════════════════════════════════════════════════════════════════════════
def deconvolve(amp):
    sm = gaussian_filter(amp.astype(float), sigma=DECONV_SIGMA)
    return np.clip(amp + DECONV_ALPHA * (amp - sm), 0, MAX_AMP_UA)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — pulse2percept simulation (both models)
# ══════════════════════════════════════════════════════════════════════════
def _load_stim(implant, amp):
    implant.stim = {
        f"E{r}{c}": float(amp[r, c])
        for r in range(ROWS) for c in range(COLS)
    }

def run_scoreboard(amp):
    """
    ScoreboardModel — idealised round Gaussian phosphenes.
    xystep=0.05 dva/px gives 2× higher spatial resolution than default 0.1
    Output: ~317×525 px
    """
    implant = build_implant()
    _load_stim(implant, amp)
    model = ScoreboardModel(
        rho           = SCOREBOARD_RHO,
        xrange        = MODEL_XRANGE,
        yrange        = MODEL_YRANGE,
        xystep        = SCOREBOARD_XYSTEP,   # ← HIGH RES
        thresh_percept= 0,
        verbose       = False
    )
    model.build()
    return model.predict_percept(implant)


def run_axonmap(amp):
    """
    AxonMapModel — biologically realistic axon streaks.
    xystep=0.06 dva/px — high res, slightly coarser than scoreboard for speed.
    λ=100µm keeps streaks short so shape stays recognisable.
    """
    implant = build_implant()
    _load_stim(implant, amp)
    model = AxonMapModel(
        rho           = AXON_RHO,
        axlambda      = AXON_LAMBDA,
        xrange        = MODEL_XRANGE,
        yrange        = MODEL_YRANGE,
        xystep        = AXON_XYSTEP,         # ← HIGH RES
        thresh_percept= 0,
        n_axons       = 500,
        n_ax_segments = 500,
        verbose       = False
    )
    model.build()
    return model.predict_percept(implant)


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6 — RENDER 6-PANEL FIGURE
# ══════════════════════════════════════════════════════════════════════════
def render(original, box, label, conf, crop,
           kpts, contour_vis, amp_raw, amp_deconv,
           percept_sb, percept_ax):

    x1, y1, x2, y2 = [int(v) for v in box]
    orig_ann = original.copy()
    cv2.rectangle(orig_ann, (x1,y1), (x2,y2), (0,255,80), 3)
    cv2.putText(orig_ann, f"{label} {conf:.0%}", (x1, max(y1-10,10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,80), 2)

    def _pdata(p):
        d = p.data.squeeze()
        return d[:,:,0] if d.ndim == 3 else d

    pd_sb = _pdata(percept_sb)
    pd_ax = _pdata(percept_ax)
    h_c, w_c = crop.shape[:2]

    fig = plt.figure(figsize=(26, 5.8), facecolor='#080808')
    gs  = gridspec.GridSpec(1, 6, figure=fig, wspace=0.22,
                            left=0.02, right=0.98, top=0.88, bottom=0.07)
    tkw = dict(color='white', fontsize=9.5, pad=5)

    # ── P1: Input + detection ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(cv2.cvtColor(orig_ann, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"Input  ·  YOLO: {label}", **tkw)
    ax1.axis('off')

    # ── P2: Contour + keypoints ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    kpy = kpts[:, 0] * (h_c-1)
    kpx = kpts[:, 1] * (w_c-1)
    ax2.scatter(kpx, kpy, c='#ff3333', s=28, zorder=5)
    ax2.plot(np.append(kpx, kpx[0]), np.append(kpy, kpy[0]),
             '-', color='#ff8888', lw=1.2, alpha=0.7)
    ax2.set_title(f"Contour  +  {len(kpts)} Keypoints\n"
                  f"(boundary → electrodes)", **tkw)
    ax2.axis('off')

    # ── P3: Raw amplitude grid ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(amp_raw, cmap='hot', vmin=0, vmax=MAX_AMP_UA, aspect='auto')
    ax3.set_title("6×10 Grid\n(current steering  µA)", **tkw)
    ax3.set_xticks(range(COLS))
    ax3.set_xticklabels([str(i+1) for i in range(COLS)], color='#ccc', fontsize=6)
    ax3.set_yticks(range(ROWS))
    ax3.set_yticklabels(list('ABCDEF'), color='#ccc', fontsize=7)
    ax3.tick_params(colors='#444', length=2)
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color='#ccc', fontsize=6)
    ax3.set_xlabel(f"{int(np.sum(amp_raw > 1))}/60 active", color='#aaa', fontsize=7)

    # ── P4: Deconvolved grid ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    im4 = ax4.imshow(amp_deconv, cmap='hot', vmin=0, vmax=MAX_AMP_UA, aspect='auto')
    ax4.set_title(f"Deconvolution pre-emphasis\n"
                  f"(α={DECONV_ALPHA}  σ={DECONV_SIGMA})", **tkw)
    ax4.set_xticks(range(COLS))
    ax4.set_xticklabels([str(i+1) for i in range(COLS)], color='#ccc', fontsize=6)
    ax4.set_yticks(range(ROWS))
    ax4.set_yticklabels(list('ABCDEF'), color='#ccc', fontsize=7)
    ax4.tick_params(colors='#444', length=2)
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    plt.setp(cb4.ax.yaxis.get_ticklabels(), color='#ccc', fontsize=6)
    ax4.set_xlabel("boosted edges → sharper percept", color='#aaa', fontsize=7)

    # ── P5: ScoreboardModel (HIGH RES) ──────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.imshow(pd_sb, cmap='gray', aspect='auto',
               interpolation='bilinear')
    ax5.set_title(f"ScoreboardModel  (idealised)\n"
                  f"ρ={SCOREBOARD_RHO}µm  ·  {pd_sb.shape[1]}×{pd_sb.shape[0]}px",
                  **tkw)
    ax5.axis('off')

    # ── P6: AxonMapModel (HIGH RES) ─────────────────────────────────
    ax6 = fig.add_subplot(gs[5])
    ax6.imshow(pd_ax, cmap='magma', aspect='auto',
               interpolation='bilinear')
    ax6.set_title(f"AxonMapModel  (realistic)\n"
                  f"ρ={AXON_RHO}µm  λ={AXON_LAMBDA}µm  ·  {pd_ax.shape[1]}×{pd_ax.shape[0]}px",
                  **tkw)
    ax6.axis('off')

    fig.text(0.5, 0.005,
        "Current Steering: bilinear amplitude split  ·  "
        "Deconvolution: unsharp-mask pre-emphasis  ·  "
        f"Resolution: xystep={SCOREBOARD_XYSTEP}dva/px (Scoreboard)  "
        f"{AXON_XYSTEP}dva/px (AxonMap)",
        ha='center', color='#555', fontsize=7)
    fig.suptitle(
        "Epiretinal Prosthesis  —  Contour Keypoints  +  "
        "Current Steering  +  Deconvolution  +  High-Res Output",
        color='white', fontsize=13, fontweight='bold', y=0.97)

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight',
                facecolor='#080808')
    plt.close(fig)
    print(f"\n  ✓ Saved → {OUTPUT_PATH}")
    subprocess.Popen(["open", OUTPUT_PATH])


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Image path: ").strip().strip("'\"")

    image_path = os.path.expanduser(image_path)
    if not os.path.isfile(image_path):
        print(f"[ERROR] Not found: {image_path}")
        sys.exit(1)

    print(f"\n[1/5] Loading YOLO...")
    yolo = YOLO("yolov8n.pt")

    print(f"[2/5] Detecting object in: {os.path.basename(image_path)}")
    original, crop, label, conf, box = detect_and_crop(image_path, yolo)
    print(f"      → '{label}'  ({conf:.0%})")

    print(f"[3/5] Extracting contour keypoints...")
    kpts, contour_vis, fg = extract_contour_keypoints(crop, n=N_KEYPOINTS)
    print(f"      → {len(kpts)} keypoints  "
          f"row=[{kpts[:,0].min():.2f}, {kpts[:,0].max():.2f}]  "
          f"col=[{kpts[:,1].min():.2f}, {kpts[:,1].max():.2f}]")

    print(f"[4/5] Current steering → amplitude grid + deconvolution...")
    amp_raw    = keypoints_to_amplitudes(kpts)
    amp_deconv = deconvolve(amp_raw)
    print(f"      → {int(np.sum(amp_raw > 1))}/60 electrodes active  "
          f"max={amp_deconv.max():.1f}µA")

    print(f"[5/5] Running pulse2percept models (high-res xystep=0.05/0.06)...")
    print(f"      → ScoreboardModel...")
    percept_sb = run_scoreboard(amp_deconv)
    print(f"      → AxonMapModel (building axon fiber map)...")
    percept_ax = run_axonmap(amp_deconv)

    def _sz(p):
        d = p.data.squeeze()
        d = d[:,:,0] if d.ndim==3 else d
        return d.shape
    print(f"      → ScoreboardModel output: {_sz(percept_sb)}")
    print(f"      → AxonMapModel output:    {_sz(percept_ax)}")

    print(f"      → Rendering 6-panel figure...")
    render(original, box, label, conf, crop,
           kpts, contour_vis, amp_raw, amp_deconv,
           percept_sb, percept_ax)
    print("Done!  retinal_percept.png is on your Desktop.")


if __name__ == "__main__":
    main()
 
"""Standalone runner mirroring circle_coordinate_extraction.ipynb.

Usage:
    source .venv/bin/activate
    python run.py input/drawing.png        # single image
    python run.py                          # batch every image in input/
    python test_line_end.py                # self-check for the line-end circles

Outputs per image: <stem>_coordinates.csv and <stem>_result.png.

Two different circle/line relations are reported, and they are NOT the same:
  HasArrow  (FR-4)  — a measurement line PASSES THROUGH this circle
  AtLineEnd (FR-10) — a measurement line TERMINATES on this circle, i.e. the
                      arrowhead lands on it. One per arrow, always inside the
                      square. Circled in black in the annotated PNG.
A circle can be both, and can even belong to one reference while being the end
of another (e.g. it sits on R2's line while R3's arrow stops on it).
"""
import os
import sys
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class Config:
    input_folder: str = "input"
    output_folder: str = "output"
    # --- ADAPTIVITY ---
    # When True, all pixel-based parameters below are auto-derived from the
    # detected dot size + square size, so the app works across resolutions
    # without manual tuning. Set False to use the fixed values as-is.
    adaptive: bool = True

    # FR-2 circle (colour-blob) detection  (px values are DEFAULTS; auto-scaled when adaptive)
    min_radius: int = 8
    max_radius: int = 45
    min_blob_area: int = 220
    min_circularity: float = 0.55   # 1.0 = perfect disc; drops arrowheads (scale-free)
    open_ksize: int = 7             # morphological opening kernel (removes lines)
    white_val: int = 180           # HSV V above this + low S == white background (scale-free)
    white_sat: int = 60
    # FR-3 arrow (colour-stroke) detection
    min_line_length: int = 120     # min extent (px) for a stroke to count as an arrow
    arrow_max_gap: int = 55        # Hough gap to jump over dots along a shaft
    arrow_angle_tol: float = 8.0   # deg: segments within this angle = same arrow (scale-free)
    arrow_rho_tol: float = 20.0    # px: and within this perpendicular offset
    arrow_extend: int = 30         # extend each arrow past its ends to reach the head
    color_group_tol: float = 14.0  # hue tol: same-colour arrows = one reference group (scale-free)
    row_bin: int = 60              # px band used to order rows top->bottom
    # FR-4 proximity
    line_tolerance: int = 10
    # FR-10 line-end (arrowhead) circle
    end_hue_tol: float = 14.0      # hue tol when isolating one arrow's own colour
    end_max_dist: float = 55.0     # px: circle centre -> arrow tip (auto-scaled: 2.5*r0)
    end_probe_radius: float = 30.0 # px: disc used to test "does the stroke continue?"
    end_decisive_ratio: float = 2.5  # tail/head mass ratio needed to trust that cue
    # FR-1 square (NO crop — mask interior only)
    roi_min_area_frac: float = 0.10
    square_margin: int = 14        # inward margin from the border line (px)
    # FR-5 reference
    reference_strategy: str = "nearest_center"
    # FR-7 calibration — the real-world size of the square along `calib_axis`.
    #   Pass per-image (CLI / API). Default 11mm width matches the sample drawing.
    #   Set square_real_mm=None to skip mm conversion (px output only).
    auto_calibrate: bool = True
    square_real_mm: Optional[float] = 11.0
    calib_axis: str = "width"      # 'width' or 'height' of the square
    calib_pixels: float = 845.0    # used only when auto_calibrate is False
    calib_mm: float = 11.0

    @property
    def mm_per_pixel(self) -> Optional[float]:
        if self.calib_pixels <= 0:
            return None
        return self.calib_mm / self.calib_pixels


@dataclass
class Circle:
    id: int
    x: int
    y: int
    r: int
    measured: bool = False
    arrow: int = 0            # 0 = no arrow; else the arrow number touching it
    at_line_end: bool = False # FR-10: an arrow TERMINATES on this circle
    end_ref: int = 0          # which reference (colour group) ends here; 0 = none
    is_reference: bool = False
    dx: float = 0.0
    dy: float = 0.0
    x_mm: float = 0.0
    y_mm: float = 0.0


Line = Tuple[int, int, int, int]


def detect_square(bgr, cfg):
    """FR-1: locate the rectangular drawing border, ANY colour. NO cropping.

    The border is the largest *rectangle-shaped* contour: it encloses a big area
    and (unlike a straight dimension line) is a closed 4-sided loop. Colour is
    irrelevant. Returns (bbox=(x,y,w,h), interior_mask) — a full-image mask that
    is 255 strictly inside the border (inward margin excludes the border line).
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # everything that is not white background (border, dots, arrows, text)
    nw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0

    def longest_run(vec):
        idx = np.where(vec)[0]
        if idx.size == 0:
            return 0, 0, 0
        parts = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        r = max(parts, key=len)
        return len(r), int(r[0]), int(r[-1])

    # The border's 4 sides are the longest continuous non-white runs. Arrows and
    # dimension lines that breach the border are shorter or sit at other offsets,
    # so run-length is robust where contours fail. Colour-agnostic.
    col_len = np.zeros(W, int); col_s = np.zeros(W, int); col_e = np.zeros(W, int)
    for x in range(W):
        col_len[x], col_s[x], col_e[x] = longest_run(nw[:, x])

    best = None
    if col_len.max() > 0.3 * H:
        # The two vertical borders are the far-apart columns with the longest
        # vertical runs. Their x positions give left/right; their run start/end
        # give top/bottom (where the frame closes) — this excludes external
        # dimension lines above/below the box, and needs no colour assumption.
        strong_c = np.where(col_len > 0.5 * col_len.max())[0]
        x0, x1 = int(strong_c.min()), int(strong_c.max())
        y0, y1 = int(np.median(col_s[strong_c])), int(np.median(col_e[strong_c]))
        if x1 - x0 > 0.3 * W and y1 - y0 > 0.2 * H:
            best = (x0, y0, x1 - x0, y1 - y0)

    mask = np.zeros((H, W), np.uint8)
    if best is None:                       # fallback: whole image
        mask[:] = 255
        return (0, 0, W, H), mask
    x, y, w, h = best
    m = max(6, int(0.012 * min(w, h)))     # inward margin scales with the square
    mask[y + m:y + h - m, x + m:x + w - m] = 255
    return best, mask


def _nonwhite_mask(bgr, cfg):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    val, sat = hsv[:, :, 2], hsv[:, :, 1]
    return (~((val > cfg.white_val) & (sat < cfg.white_sat))).astype(np.uint8) * 255


def adapt_parameters(bgr, mask, cfg):
    """Make the pipeline resolution-independent.

    Estimate the typical dot radius from a quick pass over round blobs, then
    derive every pixel-based threshold from it (and from the square size). This
    is what lets the same code handle a 600px sketch or a 4000px CAD export.
    """
    if not cfg.adaptive:
        return
    blobs = _nonwhite_mask(bgr, cfg) & mask
    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_floor = 0.00002 * mask.size            # ignore specks, relative to image size
    radii = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < area_floor:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        if r > 0 and a / (np.pi * r * r) > 0.7:  # round -> a dot (not a line/arrowhead)
            radii.append(r)
    if len(radii) < 5:                          # not enough evidence; keep defaults
        return
    r0 = float(np.median(radii))
    odd = lambda k: max(3, int(k) | 1)
    cfg.min_radius = max(3, int(0.45 * r0))
    cfg.max_radius = int(2.6 * r0)
    cfg.min_blob_area = int(0.30 * np.pi * r0 * r0)
    cfg.open_ksize = odd(round(0.55 * r0))      # > line width, < dot diameter
    cfg.min_line_length = int(3.0 * r0)
    cfg.arrow_max_gap = int(4.0 * r0)           # jump over a dot + gap along a shaft
    cfg.arrow_extend = int(1.6 * r0)
    cfg.arrow_rho_tol = max(8.0, 1.1 * r0)
    cfg.line_tolerance = max(4, int(0.55 * r0))
    cfg.row_bin = max(20, int(3.0 * r0))
    cfg.end_max_dist = 2.5 * r0                 # a dot "at the tip" is within ~2 radii
    cfg.end_probe_radius = 1.6 * r0
    return r0


def detect_circles(bgr, mask, cfg) -> List[Circle]:
    """FR-2: colour-blob detection of filled dots, restricted to `mask`.

    Robust to faint colours where Hough fails. Thin arrows are removed by the
    morphological opening; arrowheads are dropped by the circularity filter.
    Full-image coordinates (no cropping).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    val, sat = hsv[:, :, 2], hsv[:, :, 1]
    # non-white = NOT (bright AND unsaturated)
    nonwhite = ~((val > cfg.white_val) & (sat < cfg.white_sat))
    blob = (nonwhite.astype(np.uint8) * 255) & mask
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (cfg.open_ksize, cfg.open_ksize)))
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < cfg.min_blob_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (cfg.min_radius <= r <= cfg.max_radius):
            continue
        circ = a / (np.pi * r * r + 1e-6)      # 1.0 = perfect disc
        if circ < cfg.min_circularity:          # drop arrowheads / thin junk
            continue
        raw.append((int(cx), int(cy), int(round(r))))

    out = []
    for i, (x, y, r) in enumerate(sorted(raw, key=lambda t: (t[1], t[0])), start=1):
        out.append(Circle(id=i, x=x, y=y, r=r))
    return out


def detect_arrows(bgr, mask, cfg):
    """FR-3: extract each measurement arrow as a full connected stroke.

    Colour strokes (arrows) and filled dots are both saturated, so we isolate
    the THIN structures: take the coloured mask and subtract the dots (which are
    removed by a morphological opening). What remains is arrow shafts+heads.
    Each connected component = one arrow. Returns a list of uint8 masks, ordered
    top->bottom, left->right for stable numbering (A1..An).
    """
    H, W = mask.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    val, sat = hsv[:, :, 2], hsv[:, :, 1]
    colored = ((~((val > cfg.white_val) & (sat < cfg.white_sat))).astype(np.uint8) * 255) & mask
    # keep only THIN structures: subtract whatever survives an opening (= the dots)
    dots = cv2.morphologyEx(colored, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                      (cfg.open_ksize, cfg.open_ksize)))
    thin = cv2.subtract(colored, cv2.dilate(dots, np.ones((3, 3), np.uint8)))

    # Hough on the thin strokes; big gap jumps over the dots each arrow crosses
    segs = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=40,
                           minLineLength=max(20, cfg.min_line_length // 3),
                           maxLineGap=cfg.arrow_max_gap)
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4)

    # cluster segments into arrows by orientation (theta) + offset (rho)
    def theta_rho(s):
        x1, y1, x2, y2 = s
        th = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        phi = np.radians(th + 90.0)
        return th, ((x1 + x2) / 2) * np.cos(phi) + ((y1 + y2) / 2) * np.sin(phi)

    groups = []
    for s in segs:
        th, rho = theta_rho(s)
        for g in groups:
            dth = abs(th - g["th"]); dth = min(dth, 180 - dth)
            if dth < cfg.arrow_angle_tol and abs(rho - g["rho"]) < cfg.arrow_rho_tol:
                g["segs"].append(s); n = len(g["segs"])
                g["th"] = (g["th"] * (n - 1) + th) / n
                g["rho"] = (g["rho"] * (n - 1) + rho) / n
                break
        else:
            groups.append({"th": th, "rho": rho, "segs": [s]})

    # reconstruct each arrow as one clean full-length line (extended past the head)
    # and SAMPLE its real colour from the source image (so parallel same-colour
    # pairs stay their input colour instead of an invented palette colour).
    arrows = []   # list of dicts: {"mask":, "color": (B,G,R)}
    for g in groups:
        pts = np.array([[s[0], s[1]] for s in g["segs"]] + [[s[2], s[3]] for s in g["segs"]], float)
        d = np.radians(g["th"]); dirv = np.array([np.cos(d), np.sin(d)])
        proj = pts @ dirv
        p0, p1 = pts[proj.argmin()], pts[proj.argmax()]
        if np.linalg.norm(p1 - p0) < cfg.min_line_length:
            continue
        p0e = p0 - dirv * cfg.arrow_extend           # extend both ends to reach arrowheads
        p1e = p1 + dirv * cfg.arrow_extend
        m = np.zeros((H, W), np.uint8)
        cv2.line(m, tuple(np.round(p0e).astype(int)), tuple(np.round(p1e).astype(int)), 255, 3)
        m &= mask
        # sample colour: original pixels of the actual stroke under this line
        band = cv2.dilate(m, np.ones((5, 5), np.uint8)) & thin
        ys, xs = np.where(band > 0)
        if len(xs) < 10:                              # fallback: sample along the raw shaft
            band = cv2.dilate(m, np.ones((5, 5), np.uint8)) & colored
            ys, xs = np.where(band > 0)
        color = tuple(int(v) for v in np.median(bgr[ys, xs], axis=0)) if len(xs) else (0, 0, 0)
        # keep the geometry: p0/p1 are the *unclipped* stroke ends (extended past
        # the arrowhead), dir is the unit vector p0 -> p1. FR-10 needs these to
        # tell the head end from the tail end.
        arrows.append({"mask": m, "color": color,
                       "p0": p0e, "p1": p1e, "dir": dirv})

    arrows.sort(key=lambda a: (round(np.where(a["mask"] > 0)[0].mean() / cfg.row_bin),
                               np.where(a["mask"] > 0)[1].mean()))
    return arrows


def _bgr_hue(bgr):
    """OpenCV hue (0..180) of a single BGR colour."""
    px = np.uint8([[list(bgr)]])
    return int(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 0])


def group_arrows_by_color(arrows, cfg):
    """Merge arrows of the same input colour into one reference group.

    Two parallel same-colour lines (e.g. the red pair spanning two rows) mark a
    single dimension, so they share one colour and one reference number.
    Returns groups ordered top->bottom, each: {"color":(B,G,R), "idxs":[...]}.
    """
    used = [False] * len(arrows)
    groups = []
    for i, a in enumerate(arrows):
        if used[i]:
            continue
        hi = _bgr_hue(a["color"])
        idxs = [i]; used[i] = True
        for j in range(i + 1, len(arrows)):
            if used[j]:
                continue
            dh = abs(hi - _bgr_hue(arrows[j]["color"]))
            if min(dh, 180 - dh) < cfg.color_group_tol:
                idxs.append(j); used[j] = True
        cols = np.array([arrows[k]["color"] for k in idxs])
        groups.append({"color": tuple(int(v) for v in np.median(cols, axis=0)),
                       "idxs": idxs,
                       "masks": [arrows[k]["mask"] for k in idxs]})

    def top(g):
        return min(np.where(m > 0)[0].mean() for m in g["masks"])
    groups.sort(key=top)
    return groups


def flag_measured(circles, arrows, groups, cfg):
    """FR-4: mark each circle measured and record which REFERENCE (colour group)
    touches it. Closest arrow wins; the arrow's group gives the reference number.
    """
    if not arrows:
        return
    H, W = arrows[0]["mask"].shape
    arrow_group = {}
    for gnum, g in enumerate(groups, start=1):
        for idx in g["idxs"]:
            arrow_group[idx] = gnum
    dts = [cv2.distanceTransform(255 - a["mask"], cv2.DIST_L2, 3) for a in arrows]
    for c in circles:
        cy, cx = min(max(c.y, 0), H - 1), min(max(c.x, 0), W - 1)
        best_idx, best_dist = -1, float("inf")
        for idx, dt in enumerate(dts):
            d = float(dt[cy, cx])
            if d < c.r + cfg.line_tolerance and d < best_dist:
                best_dist, best_idx = d, idx
        if best_idx >= 0:
            c.measured = True
            c.arrow = arrow_group[best_idx]


def _arrow_hue_mask(bgr, color, cfg):
    """Pixels whose hue matches this arrow's own colour (hue wraps at 180)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    nonwhite = ~((val > cfg.white_val) & (sat < cfg.white_sat))
    dh = np.abs(hue - _bgr_hue(color))
    dh = np.minimum(dh, 180 - dh)
    return ((dh < cfg.end_hue_tol) & nonwhite).astype(np.uint8) * 255


def _disc_mass(src, p, radius):
    """Count of set pixels of `src` inside a disc of `radius` centred on p."""
    H, W = src.shape
    x, y = int(round(p[0])), int(round(p[1]))
    r = int(round(radius))
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x0 >= x1 or y0 >= y1:
        return 0
    win = src[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return int(np.count_nonzero(win & (((xx - x) ** 2 + (yy - y) ** 2) <= r * r)))


def _stroke_tip(stroke, origin, direction, reach, gap_allow):
    """Walk from `origin` along `direction` and return the LAST point where the
    stroke is still present — i.e. the physical tip of the arrow.

    `stroke` must be the arrow-coloured pixels with the detected dots knocked
    out, otherwise a dot that happens to share the arrow's colour keeps the walk
    going past the real tip. Gaps up to `gap_allow` (a dot diameter) are bridged
    so a dot lying on the shaft does not end the walk early.
    """
    H, W = stroke.shape
    origin = np.asarray(origin, float)
    tip, t, last_hit = origin, 0.0, 0.0
    while t <= reach:
        p = origin + direction * t
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < W and 0 <= y < H and _disc_mass(stroke, p, 4) > 0:
            tip, last_hit = p, t
        elif t - last_hit > gap_allow:
            break
        t += 2.0
    return tip


def _perp_width(mask_bool, p, normal, limit):
    """Length of the contiguous run of `mask_bool` through p, perpendicular to
    the stroke. 0 when p itself is not on the stroke."""
    H, W = mask_bool.shape

    def on(q):
        x, y = int(round(q[0])), int(round(q[1]))
        return 0 <= x < W and 0 <= y < H and bool(mask_bool[y, x])

    if not on(p):
        return 0.0
    total = 1.0
    for sgn in (1.0, -1.0):
        t = 1.0
        while t <= limit and on(np.asarray(p, float) + normal * (sgn * t)):
            total += 1.0
            t += 1.0
    return total


def _head_width(mask_bool, end_pt, inward, span, r0):
    """Widest point of the stroke in the last `span` px before `end_pt`.

    An arrowhead is a triangle several times wider than the shaft it caps, and
    it is painted in the ARROW's colour even where it overlaps a dot — so on a
    hue-isolated mask it shows up as a fat bulge right at the tip, while a plain
    stroke end stays shaft-width.
    """
    normal = np.array([-inward[1], inward[0]], float)
    widths = [_perp_width(mask_bool, np.asarray(end_pt, float) + inward * t,
                          normal, 1.5 * r0)
              for t in np.arange(0.0, span, 2.0)]
    return max(widths) if widths else 0.0


def flag_line_end_circles(bgr, circles, arrows, groups, cfg, r0=None):
    """FR-10: find the circle each arrow ENDS on (the one under the arrowhead).

    `flag_measured` only says "a line passes through this dot". This adds the
    stronger relation the drawing actually encodes: every measurement arrow
    *terminates* on one dot inside the square, and that dot is the thing being
    dimensioned.

    Per arrow:
      1. Decide which end is the head. The head is where the coloured stroke
         STOPS; the tail runs on towards its leader/dimension line. So probe a
         disc just past each reconstructed end on the arrow-coloured pixels
         (with the detected dots knocked out, so a same-colour dot near the tail
         cannot masquerade as a head) — the end with far less mass is the head.
         If neither end is decisive (both ends stop inside the drawing), fall
         back to the arrowhead's own signature: it is a triangle several times
         wider than the shaft, so the end with the fat bulge wins.
      2. Walk out to the physical tip of the stroke and take the touching circle
         that is furthest along the head direction WITHOUT sitting past the tip
         (that rejects the next dot along, which the extended line may graze).

    Sets `at_line_end` / `end_ref` on the winning circles and returns
    {arrow index -> circle id}.
    """
    if not arrows or not circles:
        return {}
    H, W = arrows[0]["mask"].shape
    arrow_group = {}
    for gnum, g in enumerate(groups, start=1):
        for idx in g["idxs"]:
            arrow_group[idx] = gnum

    # every detected dot as a filled disc: used to exclude dots from the
    # "does the stroke continue?" probe
    dot_discs = np.zeros((H, W), np.uint8)
    for c in circles:
        cv2.circle(dot_discs, (c.x, c.y), int(c.r * 1.2), 255, -1)

    r0 = r0 or float(np.median([c.r for c in circles]))
    ends = {}
    for idx, a in enumerate(arrows):
        if "p0" not in a:
            continue
        p0, p1, d = np.asarray(a["p0"], float), np.asarray(a["p1"], float), a["dir"]
        hue_mask = _arrow_hue_mask(bgr, a["color"], cfg)
        shaft_only = cv2.bitwise_and(hue_mask, cv2.bitwise_not(dot_discs)) > 0

        # circles this arrow touches (independent of flag_measured's winner-takes-all)
        dt = cv2.distanceTransform(255 - a["mask"], cv2.DIST_L2, 3)
        touch = [c for c in circles
                 if float(dt[min(max(c.y, 0), H - 1), min(max(c.x, 0), W - 1)])
                 < c.r + cfg.line_tolerance]
        if not touch:
            continue
        proj = lambda p: float(np.dot(np.asarray(p, float), d))

        # --- cue 1: the stroke stops at the head, continues at the tail
        m0 = _disc_mass(shaft_only, p0 + d * (0.6 * r0), cfg.end_probe_radius)
        m1 = _disc_mass(shaft_only, p1 - d * (0.6 * r0), cfg.end_probe_radius)
        hi_is_head = None
        if max(m0, m1) > 20 and max(m0, m1) > cfg.end_decisive_ratio * min(m0, m1):
            hi_is_head = m1 < m0
        else:
            # --- cue 2: the arrowHEAD is a bulge; a bare stroke end is not.
            # (Used when both ends stop inside the drawing, so cue 1 says
            # nothing — e.g. an arrow drawn from one dot to another.)
            hb = hue_mask > 0
            span = 4.0 * r0
            w0 = _head_width(hb, p0, d, span, r0)
            w1 = _head_width(hb, p1, -d, span, r0)
            w_max, w_min = max(w0, w1), max(min(w0, w1), 1.0)
            if w_max > 1.6 * w_min and (w_max - w_min) > 0.35 * r0:
                hi_is_head = w1 > w0
        if hi_is_head is None:                 # genuinely ambiguous -> claim nothing
            continue

        head_dir = d if hi_is_head else -d
        start = (p0 + p1) / 2.0
        reach = float(np.linalg.norm(p1 - p0)) / 2.0 + 2.0 * r0
        tip = _stroke_tip(shaft_only, start, head_dir, reach, 3.5 * r0)
        t_tip = float(np.dot(tip, head_dir))

        # the dot under the head hides most of the arrowhead, so the last
        # VISIBLE stroke pixel sits ~one dot short of the true tip: allow the
        # winning centre to sit that far past it (still far less than the
        # spacing to the next dot, which is what this rejects)
        cand = [c for c in touch
                if np.dot((c.x, c.y), head_dir) <= t_tip + 1.8 * r0
                and t_tip - np.dot((c.x, c.y), head_dir) <= cfg.end_max_dist]
        if not cand:
            continue
        end_c = max(cand, key=lambda c: float(np.dot((c.x, c.y), head_dir)))
        end_c.at_line_end = True
        end_c.end_ref = arrow_group.get(idx, 0)
        ends[idx] = end_c.id
    return ends


def select_reference(circles, square_bbox, cfg) -> Optional[Circle]:
    cand = [c for c in circles if not c.measured] or circles
    if not cand:
        return None
    x, y, w, h = square_bbox
    if cfg.reference_strategy == "nearest_center":
        cx, cy = x + w / 2, y + h / 2
        ref = min(cand, key=lambda c: (c.x - cx) ** 2 + (c.y - cy) ** 2)
    elif cfg.reference_strategy == "top_left":
        ref = min(cand, key=lambda c: c.x + c.y)
    else:
        ref = max(cand, key=lambda c: c.r)
    ref.is_reference = True
    return ref


def compute_coordinates(circles, ref, cfg):
    s = cfg.mm_per_pixel                 # None -> no calibration, leave mm as NaN
    for c in circles:
        c.dx = float(c.x - ref.x)
        c.dy = float(ref.y - c.y)
        c.x_mm = round(c.dx * s, 3) if s else float("nan")
        c.y_mm = round(c.dy * s, 3) if s else float("nan")


NO_ARROW_COLOR = (0, 180, 0)      # green  -> circle WITHOUT arrow
REF_COLOR = (0, 0, 0)             # black  -> the (0,0) reference circle
END_COLOR = (0, 0, 0)             # black ring -> circle AT THE END of a line (FR-10)


def export_csv(circles, path):
    df = pd.DataFrame([{
        "Circle": c.id, "X(px)": int(c.dx), "Y(px)": int(c.dy),
        "X(mm)": c.x_mm, "Y(mm)": c.y_mm,
        "HasArrow": "Yes" if c.measured else "No",
        "Reference#": c.arrow,                 # 0 = none, else colour-group reference number
        "AtLineEnd": "Yes" if c.at_line_end else "No",   # FR-10: arrowhead lands here
        "LineEndRef#": c.end_ref,              # which reference's line ends on it
        "Origin(0,0)": "Yes" if c.is_reference else "No",
    } for c in circles])
    df.to_csv(path, index=False)
    return df


def visualize(bgr, circles, groups, group_color, square_bbox, path):
    """Annotate the FULL image (no crop).

    - circle WITHOUT arrow -> green box
    - circle WITH arrow    -> box in that reference's REAL input colour
      (parallel same-colour pairs share one colour + one number)
    - origin (0,0)         -> black box + '(0,0)' label
    """
    canvas = bgr.copy()
    x, y, w, h = square_bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 40, 40), 1)

    # draw each reference group's arrows in its sampled input colour + number
    for gnum, g in enumerate(groups, start=1):
        col = group_color[gnum]
        allm = np.zeros(canvas.shape[:2], np.uint8)
        for m in g["masks"]:
            canvas[m > 0] = col
            allm |= m
        ys, xs = np.where(allm > 0)
        cv2.putText(canvas, f"R{gnum}", (int(xs.min()) - 34, int(ys.mean()) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    for c in circles:
        if c.is_reference:
            color = REF_COLOR
        elif c.measured:
            color = group_color.get(c.arrow, (60, 60, 60))
        else:
            color = NO_ARROW_COLOR
        cv2.rectangle(canvas, (c.x - c.r, c.y - c.r), (c.x + c.r, c.y + c.r), color, 2)
        cv2.circle(canvas, (c.x, c.y), 2, color, -1)
        label = "(0,0)" if c.is_reference else str(c.id)
        cv2.putText(canvas, label, (c.x - c.r, c.y - c.r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        if c.at_line_end:                      # FR-10: arrow TERMINATES on this dot
            col = group_color.get(c.end_ref, END_COLOR)
            cv2.circle(canvas, (c.x, c.y), c.r + 9, END_COLOR, 3)
            cv2.circle(canvas, (c.x, c.y), c.r + 13, col, 2)
            cv2.putText(canvas, f"END R{c.end_ref}", (c.x - c.r - 6, c.y + c.r + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, END_COLOR, 2, cv2.LINE_AA)
    cv2.imwrite(path, canvas)


def process_image(image_path, cfg):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    # FR-1: detect square (any colour), build interior mask (NO crop)
    bbox, mask = detect_square(img, cfg)
    # Adaptive: derive all pixel thresholds from the detected dot/square size
    r0 = adapt_parameters(img, mask, cfg)
    # FR-7 calibration: map the square's real-world size to its pixel size.
    if cfg.auto_calibrate and cfg.square_real_mm:
        span_px = bbox[2] if cfg.calib_axis == "width" else bbox[3]
        cfg.calib_pixels, cfg.calib_mm = float(span_px), float(cfg.square_real_mm)
    elif cfg.auto_calibrate and cfg.square_real_mm is None:
        cfg.calib_pixels = 0.0                     # no mm conversion requested
    circles = detect_circles(img, mask, cfg)      # FR-2
    arrows = detect_arrows(img, mask, cfg)        # FR-3: colour-stroke arrows (+ real colour)
    groups = group_arrows_by_color(arrows, cfg)   # merge same-colour parallel pairs
    flag_measured(circles, arrows, groups, cfg)   # FR-4 (records reference number)
    # FR-10: which circle does each arrow END on (arrowhead) — inside the square
    flag_line_end_circles(img, circles, arrows, groups, cfg, r0)
    # drop reference groups that touch no dots (pure dimension leaders); renumber 1..k
    used = sorted({c.arrow for c in circles if c.arrow} | {c.end_ref for c in circles if c.end_ref})
    remap = {old: new for new, old in enumerate(used, start=1)}
    groups = [groups[old - 1] for old in used]
    for c in circles:
        c.arrow = remap.get(c.arrow, 0)
        c.end_ref = remap.get(c.end_ref, 0)
    group_color = {i + 1: g["color"] for i, g in enumerate(groups)}
    ref = select_reference(circles, bbox, cfg)    # FR-5
    if ref is None:
        print(f"[warn] no circles in {image_path}")
        return
    compute_coordinates(circles, ref, cfg)        # FR-6 + FR-7
    mpp = cfg.mm_per_pixel
    scale_txt = (f"mm_per_pixel={mpp:.5f} ({cfg.square_real_mm}mm / {cfg.calib_axis})"
                 if mpp else "px only (no mm calibration)")
    r0txt = f"dot r0≈{r0:.1f}px" if r0 else "adaptive off"
    print(f"  square={bbox}  {r0txt}  ->  {scale_txt}")
    os.makedirs(cfg.output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(cfg.output_folder, f"{stem}_coordinates.csv")
    png_path = os.path.join(cfg.output_folder, f"{stem}_result.png")
    df = export_csv(circles, csv_path)
    visualize(img, circles, groups, group_color, bbox, png_path)
    print(f"\n{image_path}")
    print(f"  circles={len(circles)}  with_arrow={sum(c.measured for c in circles)}"
          f"  references(colour groups)={len(groups)}"
          f"  at_line_end={sum(c.at_line_end for c in circles)}")
    print(f"  origin (0,0) = circle #{ref.id} at pixel ({ref.x},{ref.y})")
    per = {}
    for c in circles:
        if c.arrow:
            per[c.arrow] = per.get(c.arrow, 0) + 1
    for i, g in enumerate(groups, start=1):
        b, gr, r = g["color"]
        ends = [c.id for c in circles if c.end_ref == i]
        print(f"  Reference {i}: colour(BGR)=({b},{gr},{r})  lines={len(g['masks'])}"
              f"  circles={per.get(i, 0)}"
              f"  line-end circles={ends if ends else 'none'}")
    print(f"  -> {csv_path}\n  -> {png_path}")
    print(df.to_string(index=False))


def main():
    """CLI:
        python run.py <image>              # batch input/ if omitted
        python run.py <image> <square_mm>  # real-world size of the square
        python run.py <image> <mm> height  # measure that size along the height
        python run.py <image> none         # skip mm conversion (pixels only)
    """
    cfg = Config()
    args = sys.argv[1:]
    if len(args) >= 2:                       # calibration override
        if args[1].lower() in ("none", "px", "-"):
            cfg.square_real_mm = None
        else:
            cfg.square_real_mm = float(args[1])
    if len(args) >= 3 and args[2].lower() in ("width", "height"):
        cfg.calib_axis = args[2].lower()

    if args:
        process_image(args[0], cfg)
    else:
        files = []
        for p in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(glob.glob(os.path.join(cfg.input_folder, p)))
        if not files:
            print(f"No images in {cfg.input_folder}/ — drop your drawing there.")
        for f in sorted(files):
            process_image(f, Config())


if __name__ == "__main__":
    main()

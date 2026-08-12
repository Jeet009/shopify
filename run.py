"""Standalone runner mirroring circle_coordinate_extraction.ipynb.

Usage:
    source .venv/bin/activate
    python run.py input/drawing.png        # single image
    python run.py                          # batch every image in input/
    python tests/test_line_ends.py         # synthetic checks for FR-4b

Besides flagging which dots a line merely PASSES THROUGH (FR-4), the pipeline
also reports the dot each line STOPS AT inside the square (FR-4b, `AtLineEnd`) —
arrowheaded or plain-ended, and whether the dot sits just short of or just past
the ending.
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
    # FR-4b arrow END (head) detection — the dot an arrow actually points at
    head_scan: float = 3.0         # how far back from the extended end to hunt the head (xr0)
    head_thick_ratio: float = 1.8  # head half-width must exceed shaft half-width by this
    head_margin: float = 1.3       # ambiguous when the two ends are within this ratio
    # slack (xr0) allowed BEYOND a dot's own radius: the line may stop short of
    # the dot or overshoot it. Keep below ~half the dot pitch, otherwise a line
    # dying in open space would claim the dot next to it.
    end_circle_tol: float = 1.0
    head_hue_tol: float = 12.0     # hue window used to isolate one arrow's own colour
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
    # filled in by adapt_parameters(): median dot radius, the pipeline's length unit
    dot_radius: Optional[float] = None

    @property
    def r0(self) -> float:
        """Typical dot radius in px — the scale every geometric rule is written in."""
        if self.dot_radius:
            return float(self.dot_radius)
        return max(3.0, self.max_radius / 2.6)

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
    at_line_end: bool = False  # a line stops at this dot (arrowhead or plain end)
    end_arrow: int = 0         # which reference's line ends here (0 = none)
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
    cfg.dot_radius = r0
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
    arrows = []   # list of dicts: {"mask":, "core":, "dir":, "color": (B,G,R)}
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
        # `core` is the shaft WITHOUT the extension: it stops short of the ending,
        # so anything it touches is really part of this stroke. Used by
        # detect_line_ends() to grow the stroke without swallowing the next dot.
        core = np.zeros((H, W), np.uint8)
        cv2.line(core, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 255, 3)
        core &= mask
        # sample colour: original pixels of the actual stroke under this line
        band = cv2.dilate(m, np.ones((5, 5), np.uint8)) & thin
        ys, xs = np.where(band > 0)
        if len(xs) < 10:                              # fallback: sample along the raw shaft
            band = cv2.dilate(m, np.ones((5, 5), np.uint8)) & colored
            ys, xs = np.where(band > 0)
        color = tuple(int(v) for v in np.median(bgr[ys, xs], axis=0)) if len(xs) else (0, 0, 0)
        arrows.append({"mask": m, "core": core, "dir": dirv, "color": color})

    arrows.sort(key=lambda a: (round(np.where(a["mask"] > 0)[0].mean() / cfg.row_bin),
                               np.where(a["mask"] > 0)[1].mean()))
    return arrows


def _stroke_pixels(colored, hue, mask, arrow, cfg):
    """Every pixel of ONE stroke: its own colour, inside the square, and touching
    its shaft. Returns (stroke_mask, centre, unit_dir).

    Colour + connectivity together are what make the ends trustworthy. Colour
    alone would also catch same-coloured dots lying further along the same line
    (past the ending), which would push the measured end too far. Connectivity is
    tested against `core` — the shaft *without* the extension — so the ending
    itself (arrowhead or plain stub) is included while the next dot along is not.
    """
    H, W = mask.shape
    r0 = cfg.r0
    dh = np.abs(hue - _bgr_hue(arrow["color"]))
    dh = np.minimum(dh, 180 - dh)
    cm = (colored & (dh < cfg.head_hue_tol)).astype(np.uint8)
    if cm.sum() < 50:                        # unusual hue (e.g. black line): drop the filter
        cm = colored.astype(np.uint8)

    # keep only the blobs the shaft actually runs through
    n, lab = cv2.connectedComponents(cm, connectivity=8)
    hit = np.bincount(lab[(arrow["core"] > 0) & (cm > 0)], minlength=n)
    keep = np.zeros(n, bool)
    keep[hit >= 5] = True
    keep[0] = False
    stroke = keep[lab]

    ys, xs = np.where(stroke)
    if len(xs) < 10:
        return None, None, None
    pts = np.stack([xs, ys], 1).astype(float)
    d = np.asarray(arrow["dir"], float)
    d = d / (np.linalg.norm(d) + 1e-9)
    c = pts.mean(0)
    # a same-colour blob merely brushing the shaft sideways must not widen it
    near = np.abs((pts - c) @ np.array([-d[1], d[0]])) < 2.5 * r0
    if near.sum() < 10:
        return None, None, None
    out = np.zeros((H, W), np.uint8)
    out[ys[near], xs[near]] = 255
    return out, c, d


def detect_line_ends(bgr, mask, arrows, cfg):
    """FR-4b: where does each line STOP inside the square?

    For every stroke this returns its terminals — the arrowhead tip when the line
    has one, the plain stub end when it does not. A terminal that sits on the
    square's border is dropped: there the line is only leaving the frame (its
    real end is off-drawing), it is not an ending inside the square.

    Each end is {"p": (x,y), "t": float, "c":, "d":, "arrow": idx, "head": bool},
    where `t` is the end's coordinate along the stroke direction `d`.
    """
    ends = []
    r0 = cfg.r0
    # distance to the outside of the square: ~0 exactly on the border line
    dt_edge = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    H, W = mask.shape
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    val, sat, hue = hsv[:, :, 2], hsv[:, :, 1], hsv[:, :, 0].astype(np.int16)
    colored = (~((val > cfg.white_val) & (sat < cfg.white_sat))) & (mask > 0)
    for idx, a in enumerate(arrows):
        stroke, c, d = _stroke_pixels(colored, hue, mask, a, cfg)
        if stroke is None:
            continue
        ys, xs = np.where(stroke > 0)
        pts = np.stack([xs, ys], 1).astype(float)
        t = (pts - c) @ d
        # half-width of the bare shaft, measured away from both endings
        thick = cv2.distanceTransform(stroke, cv2.DIST_L2, 5)
        tv = thick[ys, xs]
        lo, hi = t.min(), t.max()
        span = hi - lo
        mid = (t > lo + 0.25 * span) & (t < hi - 0.25 * span)
        shaft = float(np.median(tv[mid])) if mid.sum() > 20 else float(np.median(tv))
        shaft = max(shaft, 1.0)

        cand = []
        for sign, t_end in ((+1.0, hi), (-1.0, lo)):
            p_end = c + d * t_end
            # A blob thicker than the shaft just inside the terminal is an
            # arrowhead (or the dot the line dies on) — aim at its middle rather
            # than at the extreme pixel, which can overshoot past the dot.
            win = (t * sign > t_end * sign - cfg.head_scan * r0) & (t * sign <= t_end * sign)
            head = False
            if win.sum() > 5:
                k = np.argmax(np.where(win, tv, 0))
                if tv[k] > cfg.head_thick_ratio * shaft:
                    head = True
                    p_end = pts[k]
                    t_end = float(t[k])
            x, y = int(round(p_end[0])), int(round(p_end[1]))
            x, y = min(max(x, 0), W - 1), min(max(y, 0), H - 1)
            cand.append({"p": (x, y), "t": t_end, "c": c, "d": d, "arrow": idx,
                         "head": head, "thick": float(thick[y, x]),
                         "edge": float(dt_edge[y, x])})

        # Both ends thick and both plausible -> we cannot tell head from tail;
        # that is fine, an unarrowed line simply has two equal endings.
        if all(e["head"] for e in cand):
            t0, t1 = cand[0]["thick"], cand[1]["thick"]
            if max(t0, t1) < cfg.head_margin * min(t0, t1):
                for e in cand:
                    e["head"] = False

        for e in cand:
            if e["edge"] > 0.8 * r0:          # a real ending, not the border crossing
                ends.append(e)
    return ends


def flag_line_ends(circles, ends, arrow_group, cfg):
    """FR-4b: mark the circle sitting AT each line ending.

    The dot does not have to be exactly on the terminal — a line may stop just
    short of it or overshoot slightly past it — so a dot counts when it is
    roughly on the line's axis and within a dot-and-a-bit of the ending in
    either direction. Nearest to the ending wins.
    """
    r0 = cfg.r0
    for e in ends:
        c, d = e["c"], e["d"]
        nrm = np.array([-d[1], d[0]])
        best, best_gap = None, float("inf")
        for circ in circles:
            v = np.array([circ.x, circ.y], float) - c
            if abs(v @ nrm) > circ.r + cfg.line_tolerance:      # off the line's axis
                continue
            gap = abs((v @ d) - e["t"])                          # behind OR ahead
            if gap > circ.r + cfg.end_circle_tol * r0:
                continue
            if gap < best_gap:
                best, best_gap = circ, gap
        if best is not None:
            gnum = arrow_group.get(e["arrow"], 0)
            best.at_line_end = True
            best.measured = True
            best.end_arrow = gnum          # the line that STOPS here...
            if not best.arrow:             # ...which may differ from one crossing it
                best.arrow = gnum
            e["circle"] = best.id
    return ends


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


def arrow_group_map(groups):
    """arrow index -> its colour-group (reference) number, 1-based."""
    return {idx: gnum for gnum, g in enumerate(groups, start=1) for idx in g["idxs"]}


def flag_measured(circles, arrows, groups, cfg):
    """FR-4: mark each circle measured and record which REFERENCE (colour group)
    touches it. Closest arrow wins; the arrow's group gives the reference number.
    """
    if not arrows:
        return
    H, W = arrows[0]["mask"].shape
    arrow_group = arrow_group_map(groups)
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
END_COLOR = (0, 215, 255)         # amber  -> circle AT THE END of a line


def export_csv(circles, path):
    df = pd.DataFrame([{
        "Circle": c.id, "X(px)": int(c.dx), "Y(px)": int(c.dy),
        "X(mm)": c.x_mm, "Y(mm)": c.y_mm,
        "HasArrow": "Yes" if c.measured else "No",
        "AtLineEnd": "Yes" if c.at_line_end else "No",   # the dot a line stops at
        "EndOfRef#": c.end_arrow,              # which reference's line stops here (0 = none)
        "Reference#": c.arrow,                 # 0 = none, else colour-group reference number
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
        if c.at_line_end:                      # amber ring + END: a line stops here
            cv2.circle(canvas, (c.x, c.y), c.r + 7, END_COLOR, 3)
            cv2.putText(canvas, "END", (c.x - c.r, c.y + c.r + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, END_COLOR, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (c.x - c.r, c.y - c.r - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
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
    ends = detect_line_ends(img, mask, arrows, cfg)              # FR-4b
    flag_line_ends(circles, ends, arrow_group_map(groups), cfg)  # FR-4b
    # drop reference groups that touch no dots (pure dimension leaders); renumber 1..k
    used = sorted({n for c in circles for n in (c.arrow, c.end_arrow) if n})
    remap = {old: new for new, old in enumerate(used, start=1)}
    groups = [groups[old - 1] for old in used]
    for c in circles:
        c.arrow = remap.get(c.arrow, 0)
        c.end_arrow = remap.get(c.end_arrow, 0)
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
          f"  references(colour groups)={len(groups)}")
    end_ids = [c.id for c in circles if c.at_line_end]
    print(f"  line endings inside square={len(ends)}"
          f"  -> circles at a line end: {end_ids if end_ids else 'none'}")
    print(f"  origin (0,0) = circle #{ref.id} at pixel ({ref.x},{ref.y})")
    per = {}
    for c in circles:
        if c.arrow:
            per[c.arrow] = per.get(c.arrow, 0) + 1
    for i, g in enumerate(groups, start=1):
        b, gr, r = g["color"]
        print(f"  Reference {i}: colour(BGR)=({b},{gr},{r})  lines={len(g['masks'])}"
              f"  circles={per.get(i, 0)}")
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

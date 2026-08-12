"""Self-check for FR-10: "which circle sits at the END of a line?".

Run:  python test_line_end.py

Three checks, no external fixtures needed beyond input/drawing.png:

  1. synthetic drawing, arrows entering from outside the square (the common
     dimension-leader case)
  2. synthetic drawing, an arrow that starts AND ends inside the square (the
     case where the "stroke stops here" cue is useless and the arrowhead
     colour-coverage fallback has to decide)
  3. the real drawing at 100% and at 65% scale — the answer must be the same
     dots, proving the detection is resolution independent
"""
import os
import sys

import cv2
import numpy as np

from run import (Config, detect_square, adapt_parameters, detect_circles,
                 detect_arrows, group_arrows_by_color, flag_measured,
                 flag_line_end_circles)

PALETTE = [(60, 76, 231), (180, 70, 130), (80, 160, 90), (60, 160, 230),
           (200, 80, 200), (150, 140, 60), (120, 60, 190)]


def build_synthetic(path, arrows, size=(1200, 900), step=90, r=18, margin=140):
    """Grid of coloured dots inside a blue square + arrows.

    `arrows` is a list of (start_xy, end_xy, colour) in image coordinates; the
    arrowhead is drawn at `end_xy`, so the dot nearest end_xy is ground truth.
    """
    W, H = size
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (margin, margin), (W - margin, H - margin), (140, 60, 40), 4)
    centers = []
    y = margin + step // 2
    k = 0
    while y < H - margin - step // 3:
        x = margin + step // 2
        while x < W - margin - step // 3:
            cv2.circle(img, (x, y), r, PALETTE[k % len(PALETTE)], -1)
            centers.append((x, y))
            k += 1
            x += step
        y += step
    for p, q, col in arrows:
        cv2.arrowedLine(img, p, q, col, 4, cv2.LINE_AA, tipLength=0.05)
    cv2.imwrite(path, img)
    return centers


def run_pipeline(img):
    cfg = Config()
    bbox, mask = detect_square(img, cfg)
    r0 = adapt_parameters(img, mask, cfg)
    circles = detect_circles(img, mask, cfg)
    arrows = detect_arrows(img, mask, cfg)
    groups = group_arrows_by_color(arrows, cfg)
    flag_measured(circles, arrows, groups, cfg)
    flag_line_end_circles(img, circles, arrows, groups, cfg, r0)
    return circles, arrows


def nearest(circles, pt, limit):
    """The detected circle closest to `pt`, or None if none is within `limit`."""
    if not circles:
        return None
    c = min(circles, key=lambda c: (c.x - pt[0]) ** 2 + (c.y - pt[1]) ** 2)
    return c if np.hypot(c.x - pt[0], c.y - pt[1]) <= limit else None


def check(name, img, expected_pts, tol):
    """Assert the flagged line-end circles are exactly the dots at expected_pts."""
    circles, arrows = run_pipeline(img)
    got = sorted((c.x, c.y) for c in circles if c.at_line_end)
    want = []
    for p in expected_pts:
        c = nearest(circles, p, tol)
        assert c is not None, f"{name}: no circle detected near target {p}"
        want.append((c.x, c.y))
    want = sorted(want)
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'} {name}: arrows={len(arrows)} "
          f"line-end circles={got} expected={want}")
    return ok


def main():
    os.makedirs("output", exist_ok=True)
    print("FR-10 line-end circle detection")
    results = []

    # --- 1. arrows coming in from outside the square ------------------------
    targets = [(545, 275), (995, 545), (365, 545)]
    arrows = [((60, 275), (targets[0][0] + 26, targets[0][1]), (60, 76, 231)),
              ((1160, 545), (targets[1][0] - 26, targets[1][1]), (150, 140, 60)),
              ((365, 860), (targets[2][0], targets[2][1] + 26), (180, 70, 130))]
    img = build_synthetic("output/_test_outside.png", arrows)
    results.append(check("arrows from outside", cv2.imread("output/_test_outside.png"),
                         targets, 40))

    # --- 2. an arrow that starts and ends inside the square ------------------
    t = (815, 455)
    arrows = [((275, 455), (t[0] - 26, t[1]), (60, 76, 231))]
    build_synthetic("output/_test_inside.png", arrows)
    results.append(check("arrow entirely inside", cv2.imread("output/_test_inside.png"),
                         [t], 40))

    # --- 3. the real drawing, full size and downscaled -----------------------
    real = cv2.imread("input/drawing.png")
    if real is None:
        print("  SKIP real drawing (input/drawing.png missing)")
    else:
        # the eight arrowheads in input/drawing.png, by eye, in full-size px
        heads = [(920, 84), (933, 210), (583, 333), (592, 459),
                 (758, 459), (877, 529), (1103, 463), (1188, 468)]
        results.append(check("real drawing 100%", real, heads, 30))
        s = 0.65
        small = cv2.resize(real, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        results.append(check("real drawing 65%", small,
                             [(int(x * s), int(y * s)) for x, y in heads], 30))

    for f in ("output/_test_outside.png", "output/_test_inside.png"):
        if os.path.exists(f):
            os.remove(f)
    ok = all(results)
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

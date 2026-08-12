"""Regression tests for fine-pitch, densely populated drawings (e.g. BGA ball maps).

These exercise three failure modes that a coarse drawing never triggers:
  * the drawing has an inner frame (package / ball-array outline) that ENCLOSES
    every dot — nested contours must still be detected
  * dots are small (r ~ 7px), so the stroke-removal kernel has to stay wider
    than a line while still fitting inside a dot
  * the frame's four sides must not be mistaken for measurement lines
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run as R

W, H = 1500, 1150
N, BALL_R = 22, 7                     # 22 x 22 grid, ~33px pitch
PKG = (150, 40, 40)                   # package outline
FRAME = (110, 70, 30)                 # ball-array outline (encloses every ball)
BALLS = [(150, 120, 60), (60, 120, 220), (60, 160, 60), (150, 60, 150),
         (50, 50, 50), (60, 60, 200), (180, 110, 60), (140, 80, 190)]
LEADERS = [((120, 300), (3, 2), (60, 110, 220)),      # (start, (row,col) target, colour)
           ((1420, 600), (12, 18), (170, 60, 240)),
           ((700, 1120), (17, 9), (60, 200, 120)),
           ((80, 1100), (14, 5), (40, 60, 210))]
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synthetic_bga.png")


def build():
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (300, 150), (1150, 1000), PKG, 3)
    cv2.rectangle(img, (360, 205), (1090, 945), FRAME, 3)
    rng = np.random.default_rng(7)
    gx0, gy0, gx1, gy1 = 380, 225, 1070, 925
    ctr = {}
    for r in range(N):
        for c in range(N):
            p = (int(gx0 + (gx1 - gx0) * c / (N - 1)), int(gy0 + (gy1 - gy0) * r / (N - 1)))
            cv2.circle(img, p, BALL_R, BALLS[rng.integers(len(BALLS))], -1)
            ctr[(r, c)] = p
    for start, rc, col in LEADERS:
        cv2.arrowedLine(img, start, ctr[rc], col, 3, tipLength=0.05)
    cv2.imwrite(PATH, img)
    return img, ctr


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    return bool(cond)


def main():
    img, ctr = build()
    cfg = R.Config()
    bbox, mask = R.detect_square(img, cfg)
    r0 = R.adapt_parameters(img, mask, cfg)
    circles = R.detect_circles(img, mask, cfg)
    arrows = R.detect_arrows(img, mask, cfg, bbox)
    groups = R.group_arrows_by_color(arrows, cfg)
    R.flag_measured(circles, arrows, groups, cfg)
    ends = R.detect_line_ends(img, mask, arrows, cfg)
    R.flag_line_ends(circles, ends, R.arrow_group_map(groups), cfg)

    total = N * N
    ok = check("dot size measured", r0 and 4 <= r0 <= 11, f"r0={r0}")
    # every ball is a NESTED contour inside the array frame
    ok &= check("dots inside an enclosing frame are detected",
                len(circles) >= 0.98 * total, f"{len(circles)}/{total}")
    ok &= check("no wild over-detection", len(circles) <= 1.02 * total,
                f"{len(circles)}/{total}")

    def near(p, tol=12):
        return next((c for c in circles if abs(c.x - p[0]) <= tol and abs(c.y - p[1]) <= tol), None)

    missing = [rc for rc in ctr if near(ctr[rc]) is None]
    ok &= check("no ball missed", not missing, f"missing {len(missing)}")

    # the frame's sides must not become a reference group
    fh = R._bgr_hue(FRAME)
    clash = [i + 1 for i, g in enumerate(groups)
             if min(abs(R._bgr_hue(g["color"]) - fh), 180 - abs(R._bgr_hue(g["color"]) - fh)) < 10]
    ok &= check("array frame is not read as a measurement line", not clash,
                f"groups {clash} share the frame's colour")

    hit = sum(1 for _, rc, _ in LEADERS
              if (c := near(ctr[rc])) is not None and c.at_line_end)
    ok &= check("each leader's target ball is flagged AtLineEnd",
                hit == len(LEADERS), f"{hit}/{len(LEADERS)}")
    return ok


if __name__ == "__main__":
    print("dense grid / BGA")
    sys.exit(0 if main() else 1)

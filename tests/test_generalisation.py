"""Runs the pipeline over a matrix of drawing variants.

The point is that nothing here is tuned per image: resolution, ball pitch,
background tint and whether leaders carry arrowheads all change, and the same
Config must cope. Reports recall/precision against the generator's ground truth.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run as R
import fixtures

VARIANTS = [
    ("baseline",        dict()),
    ("small (0.6x)",    dict(scale=0.6)),
    ("large (1.5x)",    dict(scale=1.5)),
    ("coarse grid",     dict(n=14)),
    ("fine grid",       dict(n=32)),
    ("no arrowheads",   dict(arrowheads=False)),
    ("pink background", dict(bg=(235, 225, 250))),
    ("grey background", dict(bg=(232, 232, 232))),
    ("no labels",       dict(label_text=False)),
    ("strong tint",     dict(bg=(205, 190, 240))),
    ("tan background",  dict(bg=(170, 200, 225))),
    # a dark theme inverts the usual "dark ink on light paper" assumption
    ("dark theme",      dict(bg=(55, 50, 45), ball_colors=[
        (250, 220, 140), (140, 220, 250), (150, 250, 170), (240, 170, 250),
        (230, 230, 230), (170, 170, 255), (250, 200, 160), (210, 180, 250)])),
]


def analyse(img):
    cfg = R.Config()
    bbox, mask = R.detect_square(img, cfg)
    R.adapt_parameters(img, mask, cfg)
    circles = R.detect_circles(img, mask, cfg)
    arrows = R.detect_arrows(img, mask, cfg, bbox)
    groups = R.group_arrows_by_color(arrows, cfg)
    R.flag_measured(circles, arrows, groups, cfg)
    ends = R.detect_line_ends(img, mask, arrows, cfg)
    R.flag_line_ends(circles, ends, R.arrow_group_map(groups), cfg)
    return cfg, bbox, circles


def score(name, kw):
    img, meta = fixtures.make_bga(**kw)
    cfg, bbox, circles = analyse(img)
    truth = list(meta["centers"].values())
    tol = max(6, meta["ball_r"])

    det = np.array([[c.x, c.y] for c in circles], float) if circles else np.zeros((0, 2))
    tru = np.array(truth, float)
    if len(det):
        dist = np.linalg.norm(tru[:, None, :] - det[None, :, :], axis=2)
        found = (dist.min(1) <= tol)
        matched = (dist.min(0) <= tol)
    else:
        found = np.zeros(len(tru), bool)
        matched = np.zeros(0, bool)
    recall = found.mean() if len(tru) else 0.0
    precision = matched.mean() if len(det) else 0.0

    # every leader must be reported as ending on its target ball
    hit = 0
    for t in meta["targets"]:
        p = meta["centers"][t]
        near = [c for c in circles if abs(c.x - p[0]) <= tol and abs(c.y - p[1]) <= tol]
        hit += bool(near and near[0].at_line_end)
    n_ends = sum(c.at_line_end for c in circles)

    # calibration: the detected square should match the package outline
    pkg_w = meta["pkg"][2]
    cal_err = abs(bbox[2] - pkg_w) / pkg_w

    ok = (recall >= 0.99 and precision >= 0.97
          and hit == len(meta["targets"]) and n_ends == len(meta["targets"])
          and cal_err <= 0.02)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} balls {found.sum():4d}/{len(tru):<4d} "
          f"recall {recall:5.1%} precision {precision:5.1%}  "
          f"ends {hit}/{len(meta['targets'])} (total {n_ends})  cal err {cal_err:5.1%}  "
          f"r0={cfg.r0:.1f}")
    return ok


if __name__ == "__main__":
    print("generalisation matrix")
    allok = True
    for name, kw in VARIANTS:
        try:
            allok &= score(name, kw)
        except Exception as exc:                       # a crash is a failure too
            print(f"  FAIL  {name:<16} raised {type(exc).__name__}: {exc}")
            allok = False
    print("\nALL PASS" if allok else "\nFAILURES")
    sys.exit(0 if allok else 1)

"""Synthetic drawing generator used by the test-suite.

`make_bga()` reproduces the structure of a real BGA ball-map drawing:

  * an outer package outline and an inner ball-array outline — nested frames
    that ENCLOSE every ball
  * a fine-pitch ball grid with a realistic populated/depopulated pattern and
    many ball colours, including a near-black one
  * dimension leaders that land on a specific ball (single arrowhead), the
    thing the pipeline has to report
  * double-headed dimension arrows between extension lines, which measure a
    span and must NOT be reported as landing on a ball
  * text labels, which must not be mistaken for balls

Everything is parameterised so the same drawing can be rendered at different
resolutions, pitches, background tints and with or without arrowheads.
"""
import cv2
import numpy as np

BALL_COLORS = [(150, 120, 60), (60, 120, 220), (60, 160, 60), (150, 60, 150),
               (45, 45, 45), (60, 60, 200), (180, 110, 60), (140, 80, 190)]
PKG_COLOR = (150, 45, 45)
ARRAY_COLOR = (110, 60, 25)
DIM_COLORS = [(60, 110, 220), (170, 60, 240), (60, 200, 120), (40, 60, 210)]


def _populated(r, c, n):
    """Perimeter rings plus interior blocks — a real map is not a full grid."""
    if r < 2 or c < 2 or r >= n - 2 or c >= n - 2:
        return True
    return (3 <= r % 7 <= 5) or (3 <= c % 7 <= 5)


def make_bga(scale=1.0, n=24, bg=(255, 255, 255), arrowheads=True,
             seed=7, rotate=0.0, label_text=True, ball_colors=None):
    """Returns (image, meta). meta carries the ground truth the tests assert on."""
    S = lambda v: int(round(v * scale))
    W, H = S(1600), S(1250)
    img = np.full((H, W, 3), bg, np.uint8)

    pkg = (S(330), S(190), S(1150), S(880))          # x, y, w, h
    arr = (S(400), S(255), S(1010), S(750))
    cv2.rectangle(img, (pkg[0], pkg[1]), (pkg[0] + pkg[2], pkg[1] + pkg[3]),
                  PKG_COLOR, max(2, S(3)))
    cv2.rectangle(img, (arr[0], arr[1]), (arr[0] + arr[2], arr[1] + arr[3]),
                  ARRAY_COLOR, max(2, S(3)))

    palette = ball_colors or BALL_COLORS
    rng = np.random.default_rng(seed)
    px = arr[2] / (n + 1)
    py = arr[3] / (n + 1)
    r = max(2, int(round(0.30 * min(px, py))))
    centers = {}
    for row in range(n):
        for col in range(n):
            if not _populated(row, col, n):
                continue
            p = (int(arr[0] + px * (col + 1)), int(arr[1] + py * (row + 1)))
            cv2.circle(img, p, r, palette[rng.integers(len(palette))], -1)
            centers[(row, col)] = p

    # --- leaders that LAND on a ball (what FR-4b must report) -----------------
    targets = [(4, 3), (9, n - 4), (n - 5, 8), (n - 3, n - 7)]
    targets = [t for t in targets if t in centers]
    starts = [(S(60), S(360)), (S(1540), S(560)), (S(760), S(1210)), (S(120), S(1180))]
    lw = max(2, S(3))
    for (t, st, col) in zip(targets, starts, DIM_COLORS):
        if arrowheads:
            cv2.arrowedLine(img, st, centers[t], col, lw, tipLength=0.05)
        else:
            cv2.line(img, st, centers[t], col, lw)

    # --- double-headed span dimensions (must NOT be read as landing on a ball) -
    span_color = (90, 90, 90)
    c0, c1 = centers[(0, 5)], centers[(0, 6)]
    ytop = arr[1] - S(70)
    for cx in (c0[0], c1[0]):                        # extension lines
        cv2.line(img, (cx, ytop), (cx, arr[1] - S(10)), span_color, max(1, S(2)))
    cv2.arrowedLine(img, (c0[0] - S(45), ytop), (c0[0], ytop), span_color, lw, tipLength=0.35)
    cv2.arrowedLine(img, (c1[0] + S(45), ytop), (c1[0], ytop), span_color, lw, tipLength=0.35)
    if label_text:
        cv2.putText(img, "0.55mm", (c0[0] - S(140), ytop - S(14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, span_color, max(1, S(2)), cv2.LINE_AA)
        cv2.putText(img, "13mm package", (pkg[0] + pkg[2] + S(20), pkg[1] + pkg[3] - S(20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, PKG_COLOR, max(1, S(2)), cv2.LINE_AA)

    if rotate:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rotate, 1.0)
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=bg)
        R = M[:, :2]
        off = M[:, 2]
        centers = {k: tuple((R @ np.array(v, float) + off).round().astype(int))
                   for k, v in centers.items()}

    return img, {"centers": centers, "targets": targets, "pkg": pkg, "arr": arr,
                 "ball_r": r, "n_balls": len(centers), "scale": scale}

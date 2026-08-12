"""Synthetic checks for FR-4b: the circle at the end of a line inside the square.

Covers the cases a real drawing throws at us:
  * a plain line that just stops (no arrowhead) — both of its ends count
  * a line that stops a little SHORT of the dot
  * a line that runs a little PAST the dot
  * a line that dies in open space — no dot must be claimed
  * a line that only crosses the square — its border crossings are not endings
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run as R

W, H = 1100, 900
SQ = (120, 90, 860, 700)          # x, y, w, h
DOT_R = 18
DOT_BGR = (60, 120, 40)           # green dots
LINE_BGR = (200, 40, 40)          # blue line, clearly a different hue


def canvas():
    img = np.full((H, W, 3), 255, np.uint8)
    x, y, w, h = SQ
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 60, 20), 3)
    return img


def grid(img, rows=6, cols=9):
    """Background dots so adapt_parameters() can measure the dot size."""
    x, y, w, h = SQ
    pts = {}
    for r in range(rows):
        for c in range(cols):
            cx = x + int(w * (c + 1) / (cols + 1))
            cy = y + int(h * (r + 1) / (rows + 1))
            cv2.circle(img, (cx, cy), DOT_R, DOT_BGR, -1)
            pts[(r, c)] = (cx, cy)
    return pts


def run(img):
    cfg = R.Config()
    bbox, mask = R.detect_square(img, cfg)
    R.adapt_parameters(img, mask, cfg)
    circles = R.detect_circles(img, mask, cfg)
    arrows = R.detect_arrows(img, mask, cfg)
    groups = R.group_arrows_by_color(arrows, cfg)
    R.flag_measured(circles, arrows, groups, cfg)
    ends = R.detect_line_ends(img, mask, arrows, cfg)
    R.flag_line_ends(circles, ends, R.arrow_group_map(groups), cfg)
    return circles, ends


def at(circles, pt, tol=25):
    """The detected circle sitting at pixel `pt`, if any."""
    for c in circles:
        if abs(c.x - pt[0]) <= tol and abs(c.y - pt[1]) <= tol:
            return c
    return None


def ends_at(circles):
    return sorted((c.x, c.y) for c in circles if c.at_line_end)


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    return bool(cond)


def test_plain_line_both_ends():
    """No arrowheads at all: a bare line from dot A to dot D ends on both dots."""
    img = canvas(); pts = grid(img)
    a, d = pts[(2, 1)], pts[(2, 5)]
    cv2.line(img, a, d, LINE_BGR, 4)
    cv2.imwrite(OUT % "plain", img)
    circles, ends = run(img)
    ca, cd = at(circles, a), at(circles, d)
    inner = [pts[(2, c)] for c in (2, 3, 4)]
    ok = check("plain line: both endpoint dots flagged",
               ca and cd and ca.at_line_end and cd.at_line_end,
               f"got {ends_at(circles)}")
    ok &= check("plain line: dots the line merely crosses are NOT ends",
                all(not at(circles, p).at_line_end for p in inner))
    ok &= check("plain line: exactly 2 endings", len(ends) == 2, f"got {len(ends)}")
    return ok


def test_line_stops_short():
    """The line stops ~0.7 dot-radius before the dot — still that dot's end."""
    img = canvas(); pts = grid(img)
    a, d = pts[(1, 1)], pts[(1, 5)]
    stop = (d[0] - DOT_R - 13, d[1])
    cv2.line(img, a, stop, LINE_BGR, 4)
    cv2.imwrite(OUT % "short", img)
    circles, _ = run(img)
    cd = at(circles, d)
    return check("line stopping SHORT of the dot still flags it",
                 cd and cd.at_line_end, f"got {ends_at(circles)}")


def test_line_overshoots():
    """The line runs ~0.8 dot-radius past the dot — still that dot's end."""
    img = canvas(); pts = grid(img)
    a, d = pts[(3, 1)], pts[(3, 5)]
    stop = (d[0] + DOT_R + 15, d[1])
    cv2.line(img, a, stop, LINE_BGR, 4)
    cv2.imwrite(OUT % "over", img)
    circles, _ = run(img)
    cd, nxt = at(circles, d), at(circles, pts[(3, 6)])
    ok = check("line overshooting the dot still flags it", cd and cd.at_line_end,
               f"got {ends_at(circles)}")
    ok &= check("overshoot does not jump to the NEXT dot along", not nxt.at_line_end)
    return ok


def test_end_in_open_space():
    """A line dying midway between dots must not claim either neighbour."""
    img = canvas(); pts = grid(img)
    a, b, nxt = pts[(4, 1)], pts[(4, 4)], pts[(4, 5)]
    stop = ((b[0] + nxt[0]) // 2, b[1])
    cv2.line(img, a, stop, LINE_BGR, 4)
    cv2.imwrite(OUT % "open", img)
    circles, _ = run(img)
    return check("ending in open space claims no dot",
                 not at(circles, b).at_line_end and not at(circles, nxt).at_line_end,
                 f"got {ends_at(circles)}")


def test_crossing_line_has_no_ends():
    """A line that only passes through the square has no ending inside it."""
    img = canvas(); grid(img)
    x, y, w, h = SQ
    cv2.line(img, (10, y + 260), (W - 10, y + 260), LINE_BGR, 4)
    cv2.imwrite(OUT % "cross", img)
    circles, ends = run(img)
    return check("border crossings are not endings", len(ends) == 0, f"got {len(ends)}")


def test_arrowhead_wins_over_tail():
    """With an arrowhead, only the head end is a real ending (tail leaves the square)."""
    img = canvas(); pts = grid(img)
    d = pts[(5, 5)]
    x, y, w, h = SQ
    cv2.arrowedLine(img, (20, d[1]), (d[0] + 4, d[1]), LINE_BGR, 4, tipLength=0.04)
    cv2.imwrite(OUT % "arrow", img)
    circles, ends = run(img)
    cd = at(circles, d)
    ok = check("arrowhead dot flagged", cd and cd.at_line_end, f"got {ends_at(circles)}")
    ok &= check("arrow gives exactly 1 ending inside the square", len(ends) == 1,
                f"got {len(ends)}")
    ok &= check("that ending is recognised as a head", ends and ends[0]["head"])
    return ok


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synthetic_%s.png")
    allok = True
    for fn in (test_plain_line_both_ends, test_line_stops_short, test_line_overshoots,
               test_end_in_open_space, test_crossing_line_has_no_ends,
               test_arrowhead_wins_over_tail):
        print(fn.__name__)
        allok &= bool(fn())
    print("\nALL PASS" if allok else "\nFAILURES")
    sys.exit(0 if allok else 1)

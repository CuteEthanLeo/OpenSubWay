"""On-screen driving control panel (mouse-clickable buttons)."""

from __future__ import annotations

from .. import config
from ..sim.train import FORWARD, NEUTRAL, REVERSE

BTN_W = 118
BTN_H = 46
GAP = 8
MARGIN_BOTTOM = 18


class Button:
    __slots__ = ("id", "x0", "y0", "x1", "y1", "label", "active", "clickable", "tint")

    def __init__(self, bid, x0, y0, x1, y1, label, active, clickable, tint):
        self.id = bid
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.label = label
        self.active = active
        self.clickable = clickable
        self.tint = tint

    def contains(self, px, py):
        return self.x0 <= px <= self.x1 and self.y0 <= py <= self.y1


class ControlPanel:
    """Lays out the driving buttons and applies clicks to the train."""

    def __init__(self, train):
        self.train = train

    def _cells(self, t):
        n_thr = config.THROTTLE_NOTCHES
        n_brk = config.BRAKE_NOTCHES
        rev = t.reverser
        # (id, label, active, clickable, tint)  tint: None normal / a color
        return [
            ("rev", f"REV {t.reverser_label()}", rev != NEUTRAL, True,
             (0.25, 0.55, 0.30) if rev == FORWARD else
             (0.55, 0.30, 0.25) if rev == REVERSE else None),
            ("throttle_down", "PWR -", False, True, None),
            ("throttle_show", f"PWR {t.throttle_notch}/{n_thr}", t.throttle_notch > 0, False,
             (0.30, 0.45, 0.20) if t.throttle_notch else None),
            ("throttle_up", "PWR +", False, True, None),
            ("brake_down", "BRK -", False, True, None),
            ("brake_show", f"BRK {t.brake_notch}/{n_brk}", t.brake_notch > 0, False,
             (0.50, 0.35, 0.15) if t.brake_notch else None),
            ("brake_up", "BRK +", False, True, None),
            ("doors", "DOORS " + ("OPEN" if t.doors_open else "SHUT"), t.doors_open, True,
             (0.45, 0.40, 0.15) if t.doors_open else None),
            ("horn", "HORN", False, True, None),
        ]

    def layout(self, sw, sh):
        cells = self._cells(self.train)
        total = len(cells) * BTN_W + (len(cells) - 1) * GAP
        x = (sw - total) / 2
        y0 = sh - BTN_H - MARGIN_BOTTOM
        buttons = []
        for bid, label, active, clickable, tint in cells:
            buttons.append(Button(bid, x, y0, x + BTN_W, y0 + BTN_H,
                                   label, active, clickable, tint))
            x += BTN_W + GAP
        return buttons

    def hit_test(self, px, py, sw, sh):
        for b in self.layout(sw, sh):
            if b.clickable and b.contains(px, py):
                return b.id
        return None

    def apply(self, button_id):
        t = self.train
        if button_id == "throttle_up":
            t.throttle_up()
        elif button_id == "throttle_down":
            t.throttle_down()
        elif button_id == "brake_up":
            t.brake_up()
        elif button_id == "brake_down":
            t.brake_down()
        elif button_id == "rev":
            t.cycle_reverser()
        elif button_id == "doors":
            t.toggle_doors()
        elif button_id == "horn":
            t.horn()

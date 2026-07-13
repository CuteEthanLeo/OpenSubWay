"""Tests for the clickable control panel hit-testing and actions."""

from __future__ import annotations

from opensubway.render.panel import ControlPanel
from opensubway.sim.train import Train
from opensubway.sim.world import build_line

SW, SH = 1280, 720


def _panel():
    train = Train(build_line())
    return ControlPanel(train), train


def _center(b):
    return (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2


def test_hit_test_maps_buttons():
    panel, _ = _panel()
    by_id = {b.id: b for b in panel.layout(SW, SH)}
    for bid in ("throttle_up", "throttle_down", "brake_up", "brake_down",
                "rev", "doors", "horn"):
        cx, cy = _center(by_id[bid])
        assert panel.hit_test(cx, cy, SW, SH) == bid


def test_display_cells_not_clickable():
    panel, _ = _panel()
    by_id = {b.id: b for b in panel.layout(SW, SH)}
    cx, cy = _center(by_id["throttle_show"])
    assert panel.hit_test(cx, cy, SW, SH) is None


def test_clicks_outside_panel_miss():
    panel, _ = _panel()
    assert panel.hit_test(5, 5, SW, SH) is None


def test_apply_mutates_train():
    panel, train = _panel()
    n0 = train.throttle_notch
    panel.apply("throttle_up")
    assert train.throttle_notch == n0 + 1
    panel.apply("throttle_down")
    assert train.throttle_notch == n0

    doors0 = train.doors_open
    panel.apply("doors")            # stopped at start -> allowed
    assert train.doors_open != doors0

    panel.apply("rev")              # cycles reverser while stopped
    assert train.reverser in (-1, 0, 1)

    panel.apply("brake_up")
    assert train.brake_notch >= 1

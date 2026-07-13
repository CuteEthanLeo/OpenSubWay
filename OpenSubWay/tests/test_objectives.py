"""Headless tests for objective completion under manual driving."""

from __future__ import annotations

import _driver

from opensubway.sim.objectives import Objectives
from opensubway.sim.train import Train
from opensubway.sim.world import build_line


def test_objectives_start_incomplete():
    line = build_line()
    objs = Objectives(line)
    assert objs.completed == 0
    assert objs.total >= 4
    assert not objs.all_done()


def test_depart_objective_after_pulling_away():
    line = build_line()
    t = Train(line)
    objs = Objectives(line)
    _driver.drive_to_next_stop(t, line)
    objs.update(t)
    assert objs.items[0].done   # "Release brakes and pull away"
    assert objs.items[1].done   # "Make your first station stop"


def test_all_objectives_complete_over_a_loop():
    line = build_line()
    t = Train(line)
    objs = Objectives(line)
    _driver.drive_full_loop(t, line)
    t.passengers_carried = 5    # passengers handled by the passenger system
    objs.update(t)
    assert objs.all_done(), [o.text for o in objs.items if not o.done]


def test_objectives_are_monotonic():
    line = build_line()
    t = Train(line)
    objs = Objectives(line)
    seen = [False] * objs.total
    for _ in range(3):
        _driver.drive_to_next_stop(t, line)
        objs.update(t)
        for i, o in enumerate(objs.items):
            if o.done:
                seen[i] = True
            assert o.done or not seen[i]

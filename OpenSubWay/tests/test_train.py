"""Headless tests for the manual (player-driven) train model."""

from __future__ import annotations

import _driver

from opensubway import config
from opensubway.sim import train as train_mod
from opensubway.sim.world import build_line


def step(train, seconds):
    for _ in range(int(seconds / config.SIM_TIMESTEP)):
        train.update(config.SIM_TIMESTEP)


def test_starts_stopped_and_braked():
    t = train_mod.Train(build_line())
    assert t.is_stopped()
    assert t.speed == 0.0
    assert t.reverser == train_mod.NEUTRAL
    assert t.doors_open is True


def test_cannot_move_with_doors_open():
    t = train_mod.Train(build_line())
    t.reverser = train_mod.FORWARD   # doors still open
    t.throttle_notch = config.THROTTLE_NOTCHES
    step(t, 3.0)
    assert t.is_stopped()


def test_throttle_accelerates_forward():
    t = train_mod.Train(build_line())
    t.doors_open = False
    t.set_reverser(train_mod.FORWARD)
    t.brake_notch = 0
    t.throttle_notch = config.THROTTLE_NOTCHES
    step(t, 2.0)
    assert t.speed > 1.0
    assert t.has_departed


def test_brake_brings_to_stop():
    t = train_mod.Train(build_line())
    t.doors_open = False
    t.set_reverser(train_mod.FORWARD)
    t.brake_notch = 0
    t.throttle_notch = config.THROTTLE_NOTCHES
    step(t, 2.0)
    assert t.speed > 0
    t.throttle_notch = 0
    t.brake_notch = config.BRAKE_NOTCHES
    step(t, 5.0)
    assert t.is_stopped()


def test_reverser_moves_backward():
    t = train_mod.Train(build_line())
    t.doors_open = False
    t.set_reverser(train_mod.REVERSE)
    t.brake_notch = 0
    t.throttle_notch = config.THROTTLE_NOTCHES
    start = t.distance
    step(t, 2.0)
    assert t.speed < 0
    assert t.distance < start


def test_reverser_locked_while_moving():
    t = train_mod.Train(build_line())
    t.doors_open = False
    t.set_reverser(train_mod.FORWARD)
    t.brake_notch = 0
    t.throttle_notch = config.THROTTLE_NOTCHES
    step(t, 2.0)
    assert not t.is_stopped()
    t.set_reverser(train_mod.REVERSE)   # should be ignored while moving
    assert t.reverser == train_mod.FORWARD


def test_never_exceeds_max_speed():
    t = train_mod.Train(build_line())
    t.doors_open = False
    t.set_reverser(train_mod.FORWARD)
    t.brake_notch = 0
    t.throttle_notch = config.THROTTLE_NOTCHES
    for _ in range(int(30 / config.SIM_TIMESTEP)):
        t.update(config.SIM_TIMESTEP)
        assert abs(t.speed) <= config.TRAIN_MAX_SPEED + 1e-6


def test_stopping_at_platform_marks_visited():
    line = build_line()
    t = train_mod.Train(line)
    assert _driver.drive_to_next_stop(t, line)
    assert t.stops_made >= 1
    assert len(t.visited) >= 2


def test_full_loop_visits_all_and_counts_loop():
    line = build_line()
    t = train_mod.Train(line)
    _driver.drive_full_loop(t, line)
    assert len(t.visited) == line.n
    assert t.loops_completed >= 1

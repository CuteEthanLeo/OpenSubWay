"""Shared test helper: a simple automatic driver for the manual Train model.

Lets headless tests exercise realistic driving (accelerate, brake, stop at each
platform) without a control panel.
"""

from __future__ import annotations

from opensubway import config
from opensubway.sim.train import FORWARD, Train


def _station_targets(line, start_d):
    base = (start_d // line.total) * line.total
    out = []
    for k in (0, 1, 2):
        for s in line.stations:
            out.append(base + k * line.total + s.distance)
    return sorted(out)


def drive_to_next_stop(train: Train, line, max_seconds: float = 90.0) -> bool:
    """Accelerate to the next station node and brake to a stop there."""
    targets = [c for c in _station_targets(line, train.distance) if c > train.distance + 0.5]
    if not targets:
        return False
    target = targets[0]

    train.set_reverser(FORWARD)
    train.reverser = FORWARD
    train.doors_open = False
    train.brake_notch = 0

    full_brake = config.BRAKE_DECEL + config.ROLLING_RESISTANCE
    steps = int(max_seconds / config.SIM_TIMESTEP)
    for _ in range(steps):
        remaining = target - train.distance
        brake_dist = (train.speed * train.speed) / (2 * full_brake) if full_brake > 0 else 0.0
        if remaining <= brake_dist + 0.3:
            train.throttle_notch = 0
            train.brake_notch = config.BRAKE_NOTCHES
        else:
            train.brake_notch = 0
            train.throttle_notch = config.THROTTLE_NOTCHES
        train.update(config.SIM_TIMESTEP)
        if train.is_stopped() and train.distance >= target - config.PLATFORM_STOP_ZONE:
            return True
    return False


def drive_full_loop(train: Train, line):
    """Stop at every station once (roughly a full loop). Returns stops made."""
    for _ in range(line.n + 1):
        drive_to_next_stop(train, line)
    return train.stops_made

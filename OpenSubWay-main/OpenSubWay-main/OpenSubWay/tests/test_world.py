"""Geometry-independent checks for the expanded smooth railway."""

from __future__ import annotations

import math

import glm

from opensubway import config
from opensubway.sim.world import build_line


def test_expanded_line_keeps_all_station_knots_exact():
    line = build_line()
    assert line.n == len(config.STATIONS)
    assert line.n == 7
    assert [s.name for s in line.stations] == [row[0] for row in config.STATIONS]
    for station in line.stations:
        pos, _ = line.position_at(station.distance)
        assert glm.length(pos - station.position) < 1e-4


def test_train_heading_changes_smoothly_around_whole_route():
    line = build_line()
    previous = line.position_at(0.0)[1]
    # Fine arc samples approximate consecutive rendered frames at speed.
    for i in range(1, 2001):
        _, current = line.position_at(line.total * i / 2000.0)
        angle = math.acos(max(-1.0, min(1.0, float(glm.dot(previous, current)))))
        assert angle < math.radians(4.0)
        previous = current


def test_secondary_routes_define_real_turnout_paths():
    assert len(config.BRANCH_LINES) >= 3
    assert all(len(path) >= 3 for path in config.BRANCH_LINES)


def test_airport_line_ground_and_underground_station_split():
    line = build_line()
    assert [s.underground for s in line.stations] == [False, False, True, True, True, True, True]
    assert line.stations[0].position.y == 0.0
    assert line.stations[-1].position.y < -10.0


def test_airport_line_is_open_with_terminal_extensions():
    line = build_line()
    assert line.closed is False
    assert line.stations[0].distance > 30.0
    assert line.total - line.stations[-1].distance > 30.0


def test_operational_station_chain_matches_published_58_578_km_length():
    line = build_line()
    operational = line.stations[-1].distance - line.stations[0].distance
    # Small geometric allowance for the shallow map curves around the published
    # cumulative kilometre points; this must never regress to a short toy route.
    assert 58_500.0 <= operational <= 58_750.0

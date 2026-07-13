"""The subway line: an ordered, closed loop of stations with arc-length math."""

from __future__ import annotations

import glm

from .. import config
from .station import Station


class Line:
    """A closed polyline through station centres, with arc-length queries."""

    def __init__(self, stations: list[Station]):
        self.stations = stations
        self.points = [s.position for s in stations]
        self.n = len(self.points)

        # Segment lengths and cumulative arc-distance (looped: last -> first).
        self.seg_len: list[float] = []
        self.cum: list[float] = [0.0]
        for i in range(self.n):
            a = self.points[i]
            b = self.points[(i + 1) % self.n]
            length = glm.length(b - a)
            self.seg_len.append(length)
            self.cum.append(self.cum[-1] + length)
        self.total = self.cum[-1]

        # Record each station's arc-distance (its node is the segment start).
        for i, s in enumerate(stations):
            s.distance = self.cum[i]

    def centroid(self) -> glm.vec3:
        c = glm.vec3(0.0)
        for p in self.points:
            c += p
        return c / float(self.n)

    def position_at(self, dist: float):
        """Return (position, forward_direction) at arc-distance ``dist``."""
        if self.total <= 0:
            return glm.vec3(self.points[0]), glm.vec3(1, 0, 0)
        d = dist % self.total
        for i in range(self.n):
            if d <= self.cum[i + 1] or i == self.n - 1:
                a = self.points[i]
                b = self.points[(i + 1) % self.n]
                seg = self.seg_len[i]
                t = 0.0 if seg == 0 else (d - self.cum[i]) / seg
                pos = a + (b - a) * t
                forward = glm.normalize(b - a) if seg > 0 else glm.vec3(1, 0, 0)
                return pos, forward
        return glm.vec3(self.points[0]), glm.vec3(1, 0, 0)

    def nearest_station_index(self, dist: float) -> int:
        """Index of the station whose node is at/just passed by ``dist``."""
        d = dist % self.total
        idx = 0
        for i in range(self.n):
            if self.cum[i] <= d:
                idx = i
        return idx


def build_line() -> Line:
    """Construct the line from ``config.STATIONS``."""
    stations = [
        Station(name=name, position=glm.vec3(x, 0.0, z), color=color)
        for (name, x, z, color) in config.STATIONS
    ]
    return Line(stations)

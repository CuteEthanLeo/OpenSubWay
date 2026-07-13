"""Airport Link Line: smooth open route with ground/underground elevations."""

from __future__ import annotations

import glm

from .. import config
from .station import Station


class Line:
    """An open, arc-length sampled Catmull-Rom railway through all stations."""

    def __init__(self, stations: list[Station]):
        self.stations = stations
        self.n = len(stations)
        self.closed = bool(getattr(config, "LINE_CLOSED", False))
        controls = [glm.vec3(s.position) for s in stations]
        if len(controls) < 2:
            raise ValueError("Airport line needs at least two stations")

        # Extend beyond both terminal platform centres so a full three-car
        # consist, signals and turnouts fit without cars stacking at endpoints.
        start_dir = glm.normalize(controls[1] - controls[0])
        end_dir = glm.normalize(controls[-1] - controls[-2])
        extension = float(getattr(config, "TERMINAL_TRACK_EXTENSION", 46.0))
        route_controls = [controls[0] - start_dir * extension]
        route_controls.extend(controls)
        route_controls.append(controls[-1] + end_dir * extension)
        station_control_indices = [i + 1 for i in range(self.n)]

        samples = max(4, int(config.TRACK_CURVE_SAMPLES))
        self.points: list[glm.vec3] = []
        control_point_indices: list[int] = []
        count = len(route_controls)
        for i in range(count - 1):
            control_point_indices.append(len(self.points))
            p0 = route_controls[max(0, i - 1)]
            p1 = route_controls[i]
            p2 = route_controls[i + 1]
            p3 = route_controls[min(count - 1, i + 2)]
            segment_length = glm.length(p2 - p1)
            raw_m1 = (p2 - p0) * 0.5
            raw_m2 = (p3 - p1) * 0.5
            max_tangent = segment_length * 0.72
            m1 = (glm.normalize(raw_m1) * min(glm.length(raw_m1), max_tangent)
                  if glm.length(raw_m1) > 1e-6 else p2 - p1)
            m2 = (glm.normalize(raw_m2) * min(glm.length(raw_m2), max_tangent)
                  if glm.length(raw_m2) > 1e-6 else p2 - p1)
            for j in range(samples):
                t = j / float(samples)
                t2, t3 = t * t, t * t * t
                # Length-limited cubic Hermite tangents avoid the loops and
                # reversals that uniform Catmull-Rom creates across the real
                # line's very uneven 5-15 km station spacing.
                point = ((2.0 * t3 - 3.0 * t2 + 1.0) * p1
                         + (t3 - 2.0 * t2 + t) * m1
                         + (-2.0 * t3 + 3.0 * t2) * p2
                         + (t3 - t2) * m2)
                self.points.append(glm.vec3(point))
        control_point_indices.append(len(self.points))
        self.points.append(glm.vec3(route_controls[-1]))
        self.path_n = len(self.points)

        self.tangents: list[glm.vec3] = []
        for i in range(self.path_n):
            prev = self.points[max(0, i - 1)]
            nxt = self.points[min(self.path_n - 1, i + 1)]
            delta = nxt - prev
            self.tangents.append(
                glm.normalize(delta) if glm.length(delta) > 1e-6 else glm.vec3(1, 0, 0)
            )

        self.seg_len: list[float] = []
        self.cum: list[float] = [0.0]
        for i in range(self.path_n - 1):
            length = float(glm.length(self.points[i + 1] - self.points[i]))
            self.seg_len.append(length)
            self.cum.append(self.cum[-1] + length)
        self.total = self.cum[-1]

        for station, ci in zip(stations, station_control_indices):
            station.distance = self.cum[control_point_indices[ci]]

    def centroid(self) -> glm.vec3:
        centre = glm.vec3(0.0)
        for point in self.points:
            centre += point
        return centre / float(self.path_n)

    def position_at(self, dist: float):
        """Return position and continuously interpolated forward at distance."""
        if self.total <= 0:
            return glm.vec3(self.points[0]), glm.vec3(1, 0, 0)
        d = max(0.0, min(self.total, float(dist)))
        if d >= self.total:
            return glm.vec3(self.points[-1]), glm.vec3(self.tangents[-1])
        for i in range(self.path_n - 1):
            if d <= self.cum[i + 1]:
                a, b = self.points[i], self.points[i + 1]
                seg = self.seg_len[i]
                t = 0.0 if seg <= 1e-8 else (d - self.cum[i]) / seg
                pos = a + (b - a) * t
                tangent = self.tangents[i] * (1.0 - t) + self.tangents[i + 1] * t
                forward = (glm.normalize(tangent) if glm.length(tangent) > 1e-6
                           else glm.vec3(1, 0, 0))
                return pos, forward
        return glm.vec3(self.points[-1]), glm.vec3(self.tangents[-1])

    def nearest_station_index(self, dist: float) -> int:
        d = max(0.0, min(self.total, float(dist)))
        return min(range(self.n), key=lambda i: abs(self.stations[i].distance - d))

    def underground_points(self) -> list[glm.vec3]:
        """Contiguous tunnel alignment beginning at the portal transition."""
        selected = [glm.vec3(p) for p in self.points if p.y < -1.0]
        return selected


def build_line() -> Line:
    stations = [
        Station(
            name=name,
            position=glm.vec3(x, y, z),
            color=color,
            underground=underground,
        )
        for (name, x, y, z, color, underground) in config.STATIONS
    ]
    return Line(stations)

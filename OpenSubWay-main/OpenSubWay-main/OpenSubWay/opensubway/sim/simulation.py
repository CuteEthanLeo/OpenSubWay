"""Fixed-timestep simulation driving the train and objectives."""

from __future__ import annotations

import math

from .. import config
from .objectives import Objectives
from .passengers import PassengerSystem
from .train import Train
from .world import build_line

RED = "red"
YELLOW = "yellow"
GREEN = "green"


class Simulation:
    def __init__(self, start_index: int = 0):
        self.line = build_line()
        self.stations = self.line.stations
        self.start_index = max(0, min(len(self.stations) - 1, int(start_index)))
        self.train = Train(self.line, self.start_index)
        self.objectives = Objectives(self.line)
        self.passengers = PassengerSystem(self.line)
        spacing = config.OPPOSING_TRAIN_SPEED * config.AIRPORT_HEADWAY_SECONDS
        start = self.line.stations[-1].distance
        self.opposing_distances = [
            max(self.line.stations[0].distance, start - spacing * i)
            for i in range(config.OPPOSING_SERVICE_COUNT)
        ]
        self._accumulator = 0.0
        self.paused = False
        self.frame_events: list[str] = []   # sound/event hooks for this frame

    def reset(self):
        self.train = Train(self.line, self.start_index)
        self.objectives = Objectives(self.line)
        self.passengers = PassengerSystem(self.line)
        spacing = config.OPPOSING_TRAIN_SPEED * config.AIRPORT_HEADWAY_SECONDS
        start = self.line.stations[-1].distance
        self.opposing_distances = [
            max(self.line.stations[0].distance, start - spacing * i)
            for i in range(config.OPPOSING_SERVICE_COUNT)
        ]
        self._accumulator = 0.0

    def update(self, frame_dt: float):
        """Advance by real frame time using fixed sub-steps for stability."""
        self.frame_events = []
        if self.paused:
            return
        self._accumulator += frame_dt
        step = config.SIM_TIMESTEP
        # Cap to avoid a spiral of death after a long stall.
        max_steps = 240
        steps = 0
        while self._accumulator >= step and steps < max_steps:
            self.train.speed_limit_kmh = self.speed_limit_kmh()
            self.train.update(step)
            self.passengers.update(step, self.train)
            for i, distance in enumerate(self.opposing_distances):
                distance -= config.OPPOSING_TRAIN_SPEED * step
                if distance < self.line.stations[0].distance:
                    distance = self.line.stations[-1].distance
                self.opposing_distances[i] = distance
            self._accumulator -= step
            steps += 1
        self.frame_events = self.train.drain_events()
        self.objectives.update(self.train)

    def train_placement(self):
        """Return (model_matrix, position, forward) for the train."""
        from ..render import camera as cam

        pos, forward = self.line.position_at(self.train.distance)
        return cam.yaw_model(pos, forward), pos, forward

    @property
    def opposing_distance(self):
        return self.opposing_distances[0]

    def consist_placements(self, opposite=False, service_index=0):
        """Return independently articulated, graded and canted car poses."""
        from ..render import camera as cam
        import glm

        base = self.opposing_distances[service_index] if opposite else self.train.distance
        spacing = config.TRAIN_SIZE[0] + config.TRAIN_CAR_GAP
        out = []
        for car in range(config.TRAIN_CARS):
            # The opposite consist faces/travels toward decreasing arc distance,
            # so its following cars lie at increasing distances.
            d = base + car * spacing if opposite else base - car * spacing
            direction = -1.0 if opposite else 1.0
            half_wheelbase = config.TRAIN_SIZE[0] * 0.29

            def track_pose(sample_d):
                p, tangent = self.line.position_at(sample_d)
                if opposite:
                    perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
                    p += perp * config.OPPOSING_TRACK_OFFSET
                return p, tangent * direction

            pos, route_forward = track_pose(d)
            front, front_tangent = track_pose(d + direction * half_wheelbase)
            rear, rear_tangent = track_pose(d - direction * half_wheelbase)
            chord = front - rear
            forward = (glm.normalize(chord) if glm.length(chord) > 1e-6
                       else glm.normalize(route_forward))

            # Equilibrium cant from v^2 / (gR), clamped to the modest body roll
            # of a modern high-speed EMU.  This changes continuously through a
            # curve and removes the old rigid corner snap.
            a = glm.normalize(glm.vec3(rear_tangent.x, 0.0, rear_tangent.z))
            b = glm.normalize(glm.vec3(front_tangent.x, 0.0, front_tangent.z))
            signed_angle = math.atan2(glm.cross(a, b).y,
                                      max(-1.0, min(1.0, glm.dot(a, b))))
            curvature = signed_angle / max(1e-5, half_wheelbase * 2.0)
            velocity = (config.OPPOSING_TRAIN_SPEED if opposite
                        else abs(self.train.speed))
            roll = math.atan((velocity * velocity * curvature) / 9.81)
            roll = max(math.radians(-5.5), min(math.radians(5.5), roll))
            out.append((cam.yaw_model(pos, forward, roll), pos, forward))
        return out

    def speed_limit_kmh(self) -> float:
        """Operationally realistic station/throat/tunnel limit profile."""
        d = self.train.distance
        nearest = min(abs(d - station.distance) for station in self.stations)
        if nearest < 220.0:
            return 40.0
        if nearest < 650.0:
            return 80.0
        # Ground turnouts and the tunnel portal are approached below line speed.
        portal = self.stations[1].distance
        if abs(d - portal) < 1100.0:
            return 120.0
        return 160.0

    def signal_state(self, track: str, signal_distance: float) -> str:
        """Three-aspect block signal driven by the train ahead."""
        if track == "main":
            gap = self.train.distance - signal_distance
        else:
            gaps = [signal_distance - d for d in self.opposing_distances]
            positive = [gap for gap in gaps if gap >= 0.0]
            gap = min(positive, default=-1.0)
        block = config.SIGNAL_BLOCK_LENGTH
        if 0.0 <= gap < block:
            return RED
        if 0.0 <= gap < block * 2.0:
            return YELLOW
        return GREEN

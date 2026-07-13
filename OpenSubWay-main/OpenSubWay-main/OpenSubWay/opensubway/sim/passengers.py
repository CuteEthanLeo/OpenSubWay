"""Reactive passengers: they wait on platforms and board when doors open."""

from __future__ import annotations

import math
import random

import glm

from .. import config

WAITING = "waiting"
BOARDING = "boarding"


class Passenger:
    __slots__ = ("station", "pos", "phase", "state", "target")

    def __init__(self, station, pos, phase):
        self.station = station
        self.pos = pos            # glm.vec3 on the platform (y = 0)
        self.phase = phase        # for idle bob
        self.state = WAITING
        self.target = None        # glm.vec3 when boarding


class PassengerSystem:
    """Spawns passengers on platforms and boards them when the train stops."""

    def __init__(self, line):
        self.line = line
        self.rng = random.Random(4242)
        self.passengers: list[Passenger] = []
        self._spawn_timers = [self.rng.uniform(0, config.PASSENGER_SPAWN_INTERVAL)
                              for _ in range(line.n)]
        self._board_timer = 0.0

        # Precompute each platform's frame (centre, tangent, perpendicular).
        self.platform = []
        for i in range(line.n):
            node, tangent = line.position_at(line.stations[i].distance)
            tangent = glm.normalize(tangent)
            perp = glm.normalize(glm.vec3(-tangent.z, 0.0, tangent.x))
            # Match the centre of the rebuilt 4.2 m player-side platform.
            off = config.RAIL_HALF_WIDTH + 0.55 + 5.6 / 2.0
            centre = node + perp * off
            self.platform.append((centre, tangent, perp))

        for i in range(line.n):
            for _ in range(self.rng.randint(1, 3)):
                self._spawn(i)

    # ------------------------------------------------------------- spawning
    def _spawn(self, station: int):
        count = sum(1 for p in self.passengers if p.station == station)
        if count >= config.PASSENGER_MAX_PER_STATION:
            return
        centre, tangent, perp = self.platform[station]
        along = self.rng.uniform(-config.STATION_SIZE[0] * 0.45, config.STATION_SIZE[0] * 0.45)
        side = self.rng.uniform(-1.2, 1.2)
        pos = centre + tangent * along + perp * side
        pos = glm.vec3(pos.x, centre.y + 0.64, pos.z)
        self.passengers.append(Passenger(station, pos, self.rng.uniform(0, 6.28)))

    # -------------------------------------------------------------- update
    def update(self, dt: float, train):
        line = self.line
        for i in range(line.n):
            self._spawn_timers[i] -= dt
            if self._spawn_timers[i] <= 0.0:
                self._spawn_timers[i] += config.PASSENGER_SPAWN_INTERVAL
                self._spawn(i)

        idx = train.at_station_index()
        boarding_open = idx is not None and train.is_stopped() and train.doors_open

        train_pos, _ = line.position_at(train.distance)
        door = glm.vec3(train_pos.x, train_pos.y + 0.64, train_pos.z)

        if boarding_open:
            self._board_timer += dt
            while self._board_timer >= config.PASSENGER_BOARD_TIME:
                self._board_timer -= config.PASSENGER_BOARD_TIME
                p = self._next_waiting(idx)
                if p is None:
                    break
                p.state = BOARDING
                p.target = door
        else:
            self._board_timer = 0.0

        # Advance boarders toward the train door; idle passengers just bob.
        remaining = []
        for p in self.passengers:
            p.phase += dt * 3.0
            if p.state == BOARDING:
                tgt = door
                d = tgt - p.pos
                dist = glm.length(d)
                if dist < 0.7 or not train.is_stopped():
                    if dist < 2.5:
                        train.passengers_carried += 1
                    continue  # boarded (removed)
                step = glm.normalize(d) * min(dist, 3.5 * dt)
                p.pos = p.pos + step
            remaining.append(p)
        self.passengers = remaining

    def _next_waiting(self, station):
        for p in self.passengers:
            if p.station == station and p.state == WAITING:
                return p
        return None

    # ------------------------------------------------------------ rendering
    def instances(self):
        """Yield (position glm.vec3, bob_y) for each passenger to render."""
        out = []
        for p in self.passengers:
            bob = 0.06 * math.sin(p.phase) if p.state == WAITING else 0.0
            out.append((p.pos, bob))
        return out

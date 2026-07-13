"""Player-driven train: throttle, brake, reverser and doors (stays on rails)."""

from __future__ import annotations

from .. import config

FORWARD = 1
NEUTRAL = 0
REVERSE = -1


class Train:
    """A manually driven train.

    The player sets a **throttle** notch, a **brake** notch, a **reverser**
    (forward / neutral / reverse) and opens/closes the **doors**. Physics
    integrate a signed ``speed`` along the line's arc-distance; the train cannot
    steer (it is on rails). Progress fields (``visited`` etc.) are still exposed
    for the objectives / HUD.
    """

    def __init__(self, line, start_index: int = 0):
        self.line = line
        self.n = line.n

        self.from_idx = start_index
        self.distance = line.stations[start_index].distance  # absolute arc-distance
        self.speed = 0.0                                      # signed (units/sec)

        # Driver controls.
        self.throttle_notch = 0
        self.brake_notch = config.BRAKE_NOTCHES   # start fully braked & stopped
        self.reverser = NEUTRAL
        self.doors_open = True                    # start at Central with doors open

        # Progress tracking (read by objectives / HUD / passengers).
        self.visited = {start_index}
        self.stops_made = 0
        self.loops_completed = 0
        self.completed_runs = 0
        self._last_terminal = start_index
        self.has_departed = False
        self.passengers_carried = 0
        self.speed_limit_kmh = 160.0
        self.atp_active = False

        # One-shot events for the sound / passenger systems (consumed each frame).
        self.events: list[str] = []
        self._prev_doors = self.doors_open

    # ------------------------------------------------------------- controls
    def throttle_up(self):
        self.throttle_notch = min(config.THROTTLE_NOTCHES, self.throttle_notch + 1)

    def throttle_down(self):
        self.throttle_notch = max(0, self.throttle_notch - 1)

    def brake_up(self):
        self.brake_notch = min(config.BRAKE_NOTCHES, self.brake_notch + 1)

    def brake_down(self):
        self.brake_notch = max(0, self.brake_notch - 1)

    def set_reverser(self, direction: int):
        # Only change direction when essentially stopped.
        if self.is_stopped():
            self.reverser = max(-1, min(1, direction))

    def cycle_reverser(self):
        if self.is_stopped():
            order = [REVERSE, NEUTRAL, FORWARD]
            self.reverser = order[(order.index(self.reverser) + 1) % 3]

    def toggle_doors(self):
        if self.is_stopped():
            self.doors_open = not self.doors_open
            self.events.append("doors")

    def horn(self):
        self.events.append("horn")

    # -------------------------------------------------------------- queries
    def is_stopped(self) -> bool:
        return abs(self.speed) < config.STOP_EPSILON

    @property
    def current_station_index(self) -> int:
        return self.from_idx

    @property
    def next_station_index(self) -> int:
        direction = self.reverser if self.reverser != NEUTRAL else FORWARD
        return max(0, min(self.n - 1, self.from_idx + direction))

    def at_station_index(self):
        """Index of a station whose node we are stopped within, else None."""
        if not self.is_stopped():
            return None
        d = self.distance
        for i, s in enumerate(self.line.stations):
            gap = abs(d - s.distance)
            if gap <= config.PLATFORM_STOP_ZONE:
                return i
        return None

    def current_station_name(self) -> str:
        idx = self.at_station_index()
        if idx is not None:
            return self.line.stations[idx].name
        nxt = self._nearest_ahead()
        return f"→ {self.line.stations[nxt].name}"

    def next_station_name(self) -> str:
        return self.line.stations[self._nearest_ahead()].name

    def _nearest_ahead(self) -> int:
        """Next station in the current direction of travel (or facing)."""
        direction = self.reverser if self.reverser != NEUTRAL else FORWARD
        d = self.distance
        best_i = self.n - 1 if direction > 0 else 0
        best_gap = 1e18
        for i, s in enumerate(self.line.stations):
            delta = (s.distance - d) * direction
            if 0.1 < delta < best_gap:
                best_gap, best_i = delta, i
        return best_i

    def reverser_label(self) -> str:
        return {FORWARD: "FWD", NEUTRAL: "NEU", REVERSE: "REV"}[self.reverser]

    def speed_kmh(self) -> float:
        return abs(self.speed) * 3.6

    def status_text(self) -> str:
        if self.atp_active:
            return "ATP OVERSPEED BRAKE"
        if self.is_stopped():
            return "STOPPED (doors open)" if self.doors_open else "STOPPED"
        return "MOVING"

    # ----------------------------------------------------------------- update
    def update(self, dt: float):
        accel = 0.0

        # Traction: only with a direction selected and doors shut.
        if self.reverser != NEUTRAL and not self.doors_open and self.throttle_notch > 0:
            power = self.throttle_notch / config.THROTTLE_NOTCHES
            accel += self.reverser * power * config.TRAIN_ACCEL

        self.speed += accel * dt

        # Braking + rolling resistance oppose motion, never pushing through zero.
        decel = self.brake_notch / config.BRAKE_NOTCHES * config.BRAKE_DECEL
        decel += config.ROLLING_RESISTANCE
        self.atp_active = self.speed_kmh() > self.speed_limit_kmh + 2.0
        if self.atp_active:
            decel += config.BRAKE_DECEL * 0.85
        if self.doors_open:
            decel += config.BRAKE_DECEL  # doors open => hard hold
        if self.speed > 0:
            self.speed = max(0.0, self.speed - decel * dt)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + decel * dt)

        # Clamp to top speed.
        if self.speed > config.TRAIN_MAX_SPEED:
            self.speed = config.TRAIN_MAX_SPEED
        elif self.speed < -config.TRAIN_MAX_SPEED:
            self.speed = -config.TRAIN_MAX_SPEED

        prev_distance = self.distance
        self.distance += self.speed * dt

        minimum = self.line.stations[0].distance
        maximum = self.line.stations[-1].distance
        if self.distance <= minimum:
            self.distance = minimum
            if self.speed < 0.0:
                self.speed = 0.0
        elif self.distance >= maximum:
            self.distance = maximum
            if self.speed > 0.0:
                self.speed = 0.0

        if abs(self.speed) > config.STOP_EPSILON:
            self.has_departed = True

        self._update_progress(prev_distance)

    def _update_progress(self, prev_distance):
        # Register a stop at a platform (once per arrival).
        idx = self.at_station_index()
        if idx is not None and self.is_stopped():
            if idx not in self.visited:
                self.visited.add(idx)
            if idx != self.from_idx:
                self.from_idx = idx
                self.stops_made += 1
                self.events.append("arrive")
            if idx in (0, self.n - 1) and idx != self._last_terminal and self.has_departed:
                self.completed_runs += 1
                self.loops_completed = self.completed_runs
                self._last_terminal = idx

    # --------------------------------------------------------------- events
    def drain_events(self) -> list[str]:
        ev = self.events
        self.events = []
        return ev

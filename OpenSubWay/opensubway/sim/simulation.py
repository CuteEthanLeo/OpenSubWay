"""Fixed-timestep simulation driving the train and objectives."""

from __future__ import annotations

from .. import config
from .objectives import Objectives
from .passengers import PassengerSystem
from .train import Train
from .world import build_line


class Simulation:
    def __init__(self):
        self.line = build_line()
        self.stations = self.line.stations
        self.train = Train(self.line)
        self.objectives = Objectives(self.line)
        self.passengers = PassengerSystem(self.line)
        self._accumulator = 0.0
        self.paused = False
        self.frame_events: list[str] = []   # sound/event hooks for this frame

    def reset(self):
        self.train = Train(self.line)
        self.objectives = Objectives(self.line)
        self.passengers = PassengerSystem(self.line)
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
            self.train.update(step)
            self.passengers.update(step, self.train)
            self._accumulator -= step
            steps += 1
        self.frame_events = self.train.drain_events()
        self.objectives.update(self.train)

    def train_placement(self):
        """Return (model_matrix, position, forward) for the train."""
        from ..render import camera as cam

        pos, forward = self.line.position_at(self.train.distance)
        return cam.yaw_model(pos, forward), pos, forward

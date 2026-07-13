"""In-sim objectives ("todolist"): predicates checked against train/world state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Objective:
    text: str
    predicate: Callable  # (train, line) -> bool
    done: bool = field(default=False)


class Objectives:
    def __init__(self, line):
        self.line = line
        n = line.n
        station_names = [s.name for s in line.stations]
        market = station_names.index("Market") if "Market" in station_names else 1

        self.items = [
            Objective("Release brakes and pull away", lambda t, l: t.has_departed),
            Objective("Make your first station stop", lambda t, l: t.stops_made >= 1),
            Objective("Pick up 5 passengers", lambda t, l: t.passengers_carried >= 5),
            Objective("Call at Market station", lambda t, l: market in t.visited),
            Objective("Stop at every station", lambda t, l: len(t.visited) >= n),
            Objective("Complete a full loop", lambda t, l: t.loops_completed >= 1),
        ]

    def update(self, train):
        for obj in self.items:
            if not obj.done and obj.predicate(train, self.line):
                obj.done = True

    @property
    def completed(self) -> int:
        return sum(1 for o in self.items if o.done)

    @property
    def total(self) -> int:
        return len(self.items)

    def all_done(self) -> bool:
        return self.completed == self.total
